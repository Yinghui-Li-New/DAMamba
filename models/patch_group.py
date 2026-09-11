import torch
from .default import encode as ptv3_encode  


class PatchGroup(torch.nn.Module):

    def __init__(self, group_size, use_serialization=True, scale=10.0, depth=16,
                 enable_patch_shuffle=False):
        super().__init__()
        self.group_size = group_size
        self.use_serialization = use_serialization
        self.scale = scale
        self.depth = depth
        self.z_order = ["z", "z-trans"]  
        self.enable_patch_shuffle = enable_patch_shuffle

    def forward(self, xyz_s, xyz_t=None):

        if xyz_t is None:
            grouped, center = self._serialize_and_group(xyz_s)
            return grouped, center
        else:
            unified_min = torch.min(xyz_s.amin(dim=1, keepdim=True), xyz_t.amin(dim=1, keepdim=True))
            order = self.z_order[0] 
            grouped_s, center_s = self._serialize_and_group(xyz_s, order, coord_min=unified_min)
            grouped_t, center_t = self._serialize_and_group(xyz_t, order, coord_min=unified_min)
            return grouped_s, center_s, grouped_t, center_t

    def _serialize_and_group(self, xyz, order=None, coord_min=None):

        B, N, _ = xyz.shape
        G = N // self.group_size
        if coord_min is None:
            coord_min = xyz.amin(dim=1, keepdim=True)
        grid_coord = ((xyz - coord_min) * self.scale).int().reshape(-1, 3)
        code_all = []
        for b in range(B):
            grid_b = grid_coord[b * N : (b + 1) * N]
            use_order = order if order is not None else self.z_order[0]
            code_b = ptv3_encode(grid_b, batch=None, depth=self.depth, order=use_order)
            code_all.append(code_b)
        code = torch.stack(code_all, dim=0)  # [B, N]
        sort_idx = code.argsort(dim=1)       # [B, N]
        xyz_sorted = torch.gather(xyz, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))
        grouped = xyz_sorted[:, :G * self.group_size, :].reshape(B, G, self.group_size, 3)
        if self.training and self.enable_patch_shuffle:
            perm = torch.randperm(G, device=xyz.device)
            grouped = grouped[:, perm, :]
        center = grouped.mean(dim=2)
        return grouped, center




