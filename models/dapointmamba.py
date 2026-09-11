import math
import random
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from mamba_ssm.modules.mamba_simple import Mamba
from knn_cuda import KNN
from .block import Block
from .build import MODELS
from .utils import MLP_Res, fps_subsample
from .SPD import SPD
from torch.nn import MSELoss
from .patch_group import PatchGroup

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# --- Spatial SSM ---
class SpatialSSM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, x_s, x_t):
        # x_: (B, D, G)
        ds = self.dw(x_s)
        dt = self.dw(x_t)
        w  = self.cos(ds, dt).unsqueeze(1)   # (B,1,G)
        return x_s * w, x_t * w
    
def feature_perturbation(x, epsilon=0.1):
    if random.random() < 0.5:  
        noise = torch.randn_like(x) * epsilon
        return x + noise
    return x
   
class ChannelSSM(nn.Module):
    def __init__(self, dim, segments=4):
        super().__init__()
        self.seg = segments
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
        self.alignment_strength = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_s, x_t):
        B, D, G = x_s.shape
        

        global_s = x_s.mean(dim=2)  # [B, D]
        global_t = x_t.mean(dim=2)
        strength_input = torch.cat([global_s, global_t], dim=1)
        alignment_strength = self.alignment_strength(strength_input)  # [B, 1]

        s_chunks = x_s.chunk(self.seg, 1)
        t_chunks = x_t.chunk(self.seg, 1)
        
        x_mix = torch.cat([s_chunks[0], t_chunks[1], s_chunks[2], t_chunks[3]], 1)
        x_t_mix = torch.cat([t_chunks[0], s_chunks[1], t_chunks[2], s_chunks[3]], 1)

        w = self.cos(x_mix, x_t_mix).unsqueeze(-1)  # [B, G, 1]

        adaptive_w = w * alignment_strength.unsqueeze(-1)  # [B, G, 1]
        
        return x_s * adaptive_w, x_t * adaptive_w

    
class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=256, num_pc=128):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  
        completion = self.mlp_4(x3)  
        return completion

class Encoder(nn.Module):  
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256,512,1),

        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape 
        point_groups = point_groups.reshape(bs * g, n, 3) 
        feature = self.first_conv(point_groups.transpose(2, 1))  
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  
        feature = self.second_conv(feature) 
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  
        return feature_global.reshape(bs, g, self.encoder_channel)



def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, 
                residual_in_fp32=False, fused_add_norm=False, layer_idx=None, 
                drop_path=0., device=None, dtype=None):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32 

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm #False
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        ) 

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        ) 

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        ) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        } 

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states) 
        if not self.fused_add_norm: 
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else: 
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


class MambaDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, config=None):
        super().__init__()
        if hasattr(config, "use_external_dwconv_at_last"):
            self.use_external_dwconv_at_last = config.use_external_dwconv_at_last
        else: #False
            self.use_external_dwconv_at_last = False
        self.blocks = MixerModel(d_model=embed_dim,
                                 n_layer=depth,
                                 rms_norm=config.rms_norm,
                                 drop_path=config.drop_path) 

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x
    

@MODELS.register_module()
class DAPointMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[DAPointMamba] ', logger='DAPointMamba')
        self.config = config
        self.trans_dim = config.mamba_config.trans_dim
        self.points = config.points
        num_p0 = config.num_p0
        dim_feat = config.num_p0
        num_pc = config.num_pc
        radius = config.radius
        bounding = config.bounding
        up_factors = config.up_factors
        self.order_mode = config.order_mode


        self.encoder_dims = config.mamba_config.encoder_dims
        self.depth = config.mamba_config.depth
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )


        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.config.rms_norm)
                                 
        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

        self.norm = nn.LayerNorm(self.trans_dim)
        ##
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.mamba_config.decoder_depth
        self.MAE_decoder = MambaDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            config=config,
        )

        print_log(f'[DAPointMamba] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='DAPointMamba')


        self.group_divider = PatchGroup(
            group_size=self.group_size,
            use_serialization=True,
            scale=15.0,
            depth=16,
            enable_patch_shuffle=False
        )

        self.decoder = Decoder(dim_feat=self.trans_dim, num_pc=num_pc, num_p0=num_p0, radius=radius, 
                               bounding=bounding, up_factors=up_factors)
        
        # SSM module & Alignment
        D = self.trans_dim
        self.spatial_ssm = SpatialSSM(dim=D)
        self.channel_ssm = ChannelSSM(dim=D, segments=4)
        self.mse         = MSELoss()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)    

    def forward(self, pts_s, pts_t=None,**kwargs):

        

        if pts_t is not None:
            neighborhoods_s, centers_s, neighborhoods_t, centers_t = \
                self.group_divider(pts_s, pts_t)
            
            group_input_tokens_s = self.encoder(neighborhoods_s)  # [B, G, C]
            group_input_tokens_t = self.encoder(neighborhoods_t)  # [B, G, C]
            pos_s = self.pos_embed(centers_s)
            pos_t = self.pos_embed(centers_t)
            x_s = self.drop_out(group_input_tokens_s)
            x_t = self.drop_out(group_input_tokens_t)
            
            sp_s, sp_t = self.spatial_ssm(
                x_s.permute(0,2,1), 
                x_t.permute(0,2,1)
            ) 
            
            x_s = sp_s.permute(0,2,1)   
            x_t = sp_t.permute(0,2,1)
        else:
            neighborhoods_s, centers_s = self.group_divider(pts_s)
            group_input_tokens_s = self.encoder(neighborhoods_s)  
            pos_s = self.pos_embed(centers_s)
            x_s = self.drop_out(group_input_tokens_s)
            
            x_t = None
            sp_s = sp_t = None

        out_s = self.MAE_decoder(x_s, pos_s)
        out_s = self.norm(out_s) #[32,64,384]

        if x_t is not None:
            out_t = self.MAE_decoder(x_t, pos_t)
            out_t = self.norm(out_t)

            ch_s, ch_t = self.channel_ssm(
                out_s.permute(0,2,1),  # (B,D,G)
                out_t.permute(0,2,1)
            )
            out_s = ch_s.permute(0,2,1)  # (B,G,D)
            out_t = ch_t.permute(0,2,1)

            loss_sp = self.mse(sp_s, sp_t)
            loss_ch = self.mse(ch_s, ch_t)

        else:
            loss_sp = None
            loss_ch = None

        global_feat_s = torch.max(out_s, dim=1, keepdim=True)[0] 
        global_feat = global_feat_s.transpose(1,2)
        rebuild_points = self.decoder(global_feat, pts_s)
        return rebuild_points, loss_sp, loss_ch
    
class Decoder(nn.Module):
    def __init__(self, dim_feat=256, num_pc=128, num_p0=256,
                 radius=1, bounding=True, up_factors=None):
        super(Decoder, self).__init__()
        self.num_p0 = num_p0
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors

        uppers = []
        for i, factor in enumerate(up_factors): 
            uppers.append(SPD
            (dim_feat=dim_feat, up_factor=factor, i=i, bounding=bounding, radius=radius))

        self.uppers = nn.ModuleList(uppers)
        self.ln = nn.LayerNorm(dim_feat)


    def forward(self, feat, partial):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd = []
        feat = self.ln(feat.transpose(1, 2)).transpose(1, 2) 
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  
        arr_pcd.append(pcd)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)  
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()#[32,3,512]
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, feat, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd
                          