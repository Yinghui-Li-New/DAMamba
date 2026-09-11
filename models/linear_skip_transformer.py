# models/linear_skip_transformer.py
import torch, torch.nn as nn
from .utils import MLP_Res, grouping_operation, query_knn

class LinearSkipTransformer(nn.Module):

    def __init__(self, in_channel, dim=256, n_knn=16,
                 pos_hidden_dim=64, agg='mean'):
        super().__init__()
        self.n_knn = n_knn
        self.agg = agg     
    
        self.mlp_v = MLP_Res(in_dim=in_channel*2,
                             hidden_dim=in_channel,
                             out_dim=in_channel)
        self.lin_proj = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )
        self.local_mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    @torch.cuda.amp.autocast(False)   #
    def forward(self, pos, key, query, include_self=True):

        value = self.mlp_v(torch.cat([key, query], dim=1))  
        identity = value                                     

        v = self.lin_proj(value)                            


        B, dim, N = v.shape
        xyz = pos.permute(0,2,1).contiguous()              
        idx_knn = query_knn(self.n_knn, xyz, xyz,
                            include_self=include_self)       # (B,N,n_knn)

        v_group = grouping_operation(v, idx_knn)             # (B,dim,N,n_knn)

        pos_rel = pos.view(B, 3, N, 1) - grouping_operation(pos, idx_knn)
        pos_enc = self.pos_mlp(pos_rel)                      # (B,dim,N,n_knn)

        feat = self.local_mlp(v_group + pos_enc)             # (B,dim,N,n_knn)

        if self.agg == 'mean':
            agg = feat.mean(dim=-1)                          # (B,dim,N)
        else:
            agg = feat.max(dim=-1)[0]

        out = self.conv_end(agg)                             # (B,C,N)
        return out + identity
