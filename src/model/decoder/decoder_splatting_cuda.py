from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import torchvision

from ..types import Gaussians
# from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput
from math import sqrt 
from gsplat import rasterization

from ...misc.utils import vis_depth_map

from jaxtyping import Int

DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]

@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]
    
    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def rendering_fn(
        self,
        gaussians: Gaussians,
        gaussians_dynamic: Gaussians | None,
        dynamic_view_indices: Int[Tensor, "batch N_dynamic_total"] | None,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:
        B, V, _, _  = intrinsics.shape
        H, W = image_shape
        rendered_imgs, rendered_depths, rendered_alphas = [], [], []
        
        xyzs_static = gaussians.means  # (B, N_static, 3)
        opacities_static = gaussians.opacities  # (B, N_static)
        scales_static = gaussians.scales  # (B, N_static, 3)
        rotations_static = gaussians.rotations  # (B, N_static, 4) - quaternions
        features_static = gaussians.harmonics.permute(0, 1, 3, 2).contiguous()  # (B, N_static, 3, d_sh) - spherical harmonics
        covariances_static = gaussians.covariances  # (B, N_static, 3, 3)
        


        for i in range(B):
            # Static Gaussians for this batch
            xyz_static_i = xyzs_static[i].float()  # (N_static, 3)
            opacity_static_i = opacities_static[i].squeeze(-1).float()  # (N_static,)
            scale_static_i = scales_static[i].float()  # (N_static, 3)
            rotation_static_i = rotations_static[i].float()  # (N_static, 4)
            feature_static_i = features_static[i].float()  # (N_static, 3, d_sh)
            covar_static_i = covariances_static[i].float() if covariances_static is not None else None  # (N_static, 3, 3)
            
            
            # Dynamic Gaussians for this batch (if exist)
            if gaussians_dynamic is not None and dynamic_view_indices is not None:
                xyz_dynamic_i = gaussians_dynamic.means[i].float()  # (N_dynamic, 3)
                opacity_dynamic_i = gaussians_dynamic.opacities[i].squeeze(-1).float()  # (N_dynamic,)
                scale_dynamic_i = gaussians_dynamic.scales[i].float()  # (N_dynamic, 3)
                rotation_dynamic_i = gaussians_dynamic.rotations[i].float()  # (N_dynamic, 4)
                feature_dynamic_i = gaussians_dynamic.harmonics[i].permute(0, 2, 1).contiguous()  # (N_dynamic, 3, d_sh)
                covar_dynamic_i = gaussians_dynamic.covariances[i].float() if gaussians_dynamic.covariances is not None else None  # (N_dynamic, 3, 3)
                view_idx_i = dynamic_view_indices[i]  # (N_dynamic,) - view index for each Gaussian

               
            else:
                xyz_dynamic_i = None
                view_idx_i = None
            
            test_w2c_i = extrinsics[i].float().inverse() # (V, 4, 4)
            test_intr_i_normalized = intrinsics[i].float()
            # Denormalize the intrinsics into standred format
            test_intr_i = test_intr_i_normalized.clone()
            test_intr_i[:, 0] = test_intr_i_normalized[:, 0] * W
            test_intr_i[:, 1] = test_intr_i_normalized[:, 1] * H
            sh_degree = (int(sqrt(feature_static_i.shape[-2])) - 1)

            rendering_list = []
            rendering_depth_list = []
            rendering_alpha_list = []
            
            for j in range(V):
                
                # Render dynamic Gaussians for this view only
                if xyz_dynamic_i is not None:
                    # Extract dynamic Gaussians belonging to view j
                    view_mask = (view_idx_i == j)  # (N_dynamic,)

                    if view_mask.any():
                        # Extract attributes for this view
                        xyz_dynamic_view_j = xyz_dynamic_i[view_mask]  # (N_view_j, 3)
                        opacity_dynamic_view_j = opacity_dynamic_i[view_mask]  # (N_view_j,)
                        scale_dynamic_view_j = scale_dynamic_i[view_mask]  # (N_view_j, 3)
                        rotation_dynamic_view_j = rotation_dynamic_i[view_mask]  # (N_view_j, 4)
                        feature_dynamic_view_j = feature_dynamic_i[view_mask]  # (N_view_j, 3, d_sh)
                        covar_dynamic_view_j = covar_dynamic_i[view_mask] if covar_dynamic_i is not None else None  # (N_view_j, 3, 3)
                        
                        xyz_combined = torch.cat([xyz_static_i, xyz_dynamic_view_j], dim=0)
                        opacity_combined = torch.cat([opacity_static_i, opacity_dynamic_view_j], dim=0)
                        scale_combined = torch.cat([scale_static_i, scale_dynamic_view_j], dim=0)
                        rotation_combined = torch.cat([rotation_static_i, rotation_dynamic_view_j], dim=0)
                        feature_combined = torch.cat([feature_static_i, feature_dynamic_view_j], dim=0)
                        covar_combined = torch.cat([covar_static_i, covar_dynamic_view_j], dim=0) # if covar_dynamic_view_j is not None else covar_static_i
                    else:
                        # No dynamic Gaussians for this view, use static only
                        xyz_combined = xyz_static_i
                        opacity_combined = opacity_static_i
                        scale_combined = scale_static_i
                        rotation_combined = rotation_static_i
                        feature_combined = feature_static_i
                        covar_combined = covar_static_i

                    rendering_combined, alpha_combined, _ = rasterization(
                        xyz_combined, rotation_combined, scale_combined,
                        opacity_combined, feature_combined,
                        test_w2c_i[j:j+1], test_intr_i[j:j+1], W, H,
                        sh_degree=sh_degree,
                        render_mode="RGB+D", packed=False,
                        near_plane=1e-10,
                        backgrounds=torch.zeros_like(self.background_color).unsqueeze(0),  # Transparent bg
                        radius_clip=0.1,
                        covars=covar_combined,
                        rasterize_mode='classic'
                    )
                    rendering_img_c, rendering_depth_c = torch.split(rendering_combined, [3, 1], dim=-1)
                    final_rgb = rendering_img_c
                    final_alpha = alpha_combined
                    final_depth = rendering_depth_c

                    
                else:
                    # No dynamic Gaussians at all
                    # Render static Gaussians only
                    rendering_static, alpha_static, _ = rasterization(
                        xyz_static_i, rotation_static_i, scale_static_i, opacity_static_i, feature_static_i,
                        test_w2c_i[j:j+1], test_intr_i[j:j+1], W, H,
                        sh_degree=sh_degree,
                        render_mode="RGB+D", packed=False,
                        near_plane=1e-10,
                        backgrounds=self.background_color.unsqueeze(0).repeat(1, 1),
                        radius_clip=0.1,
                        covars=covar_static_i,
                        rasterize_mode='classic'
                    )
                    rendering_img_s, rendering_depth_s = torch.split(rendering_static, [3, 1], dim=-1)
                    final_rgb = rendering_img_s
                    final_alpha = alpha_static
                    final_depth = rendering_depth_s


                final_rgb = final_rgb.clamp(0.0, 1.0)
                rendering_list.append(final_rgb.permute(0, 3, 1, 2))
                rendering_depth_list.append(final_depth)
                rendering_alpha_list.append(final_alpha)
            
            rendered_imgs.append(torch.cat(rendering_list, dim=0))
            rendered_depths.append(torch.cat(rendering_depth_list, dim=0).squeeze())
            rendered_alphas.append(torch.cat(rendering_alpha_list, dim=0).squeeze())
        
        return DecoderOutput(
            torch.stack(rendered_imgs),
            torch.stack(rendered_depths),
            torch.stack(rendered_alphas),
            lod_rendering=None
        )
        
    def forward(
        self,
        gaussians: Gaussians,
        gaussians_dynamic: Gaussians | None,
        dynamic_view_indices: Int[Tensor, "batch N_dynamic_total"] | None,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:

        return self.rendering_fn(gaussians, gaussians_dynamic, dynamic_view_indices, extrinsics, intrinsics, near, far, image_shape, depth_mode, cam_rot_delta, cam_trans_delta)

