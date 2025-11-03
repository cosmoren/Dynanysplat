from einops import rearrange
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
# import dust3r.utils.path_to_croco
from .dpt_block import DPTOutputAdapter, Interpolate, make_fusion_block
from src.model.encoder.vggt.heads.dpt_head import DPTHead
from .head_modules import UnetExtractor, AppearanceTransformer, _init_weights
from .postprocess import postprocess

class DPT_DYNAMIC_Head(DPTHead):
    def __init__(self, 
            dim_in: int,
            patch_size: int = 14,
            output_dim: int = 2,
            activation: str = "inv_log",
            conf_activation: str = "expp1",
            features: int = 128,
            out_channels: List[int] = [256, 512, 1024, 1024],
            intermediate_layer_idx: List[int] = [4, 11, 17, 23],
            pos_embed: bool = True,
            feature_only: bool = False,
            down_ratio: int = 1,
    ):
        super().__init__(dim_in, patch_size, output_dim, activation, conf_activation, features, out_channels, intermediate_layer_idx, pos_embed, feature_only, down_ratio)
        head_features_1 = features
        head_features_2 = 32
        self.input_merger = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, head_features_1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        conv2_in_channels = head_features_1 // 2

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )
    def scratch_forward(self, features: List[torch.Tensor], pts, H, W) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1


        direct_pts_feat = self.input_merger(pts)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        out = out + direct_pts_feat

        out = self.scratch.output_conv1(out)
        return out

    def forward(self, encoder_tokens: List[torch.Tensor], pts, patch_start_idx: int = 5, image_size=None, conf=None, frames_chunk_size: int = 8):
        # H, W = input_info['image_size']
        B, S, H, W, _ = pts.shape
        image_size = self.image_size if image_size is None else image_size
    
        # If frames_chunk_size is not specified or greater than S, process all frames at once
        print(f"pts.shape in dpt_dynamic_head.forward: {pts.shape}")
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(encoder_tokens, pts, patch_start_idx)


        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            chunk_output = self._forward_impl(
                encoder_tokens, pts, patch_start_idx, frames_start_idx, frames_end_idx
            )
            all_preds.append(chunk_output)
        
        # Concatenate results along the sequence dimension
        return torch.cat(all_preds, dim=1)

    def _forward_impl(self, encoder_tokens: List[torch.Tensor], pts, patch_start_idx: int = 5, frames_start_idx: int = None, frames_end_idx: int = None):
        
        if frames_start_idx is not None and frames_end_idx is not None:
            pts = pts[:, frames_start_idx:frames_end_idx]

        B, S, H, W, _ = pts.shape
        pts = pts.flatten(0, 1).permute(0, 3, 1, 2)
        patch_h, patch_w = H // self.patch_size[0], W // self.patch_size[1]

        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            # x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            if len(encoder_tokens) > 10:
                x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            else:
                list_idx = self.intermediate_layer_idx.index(layer_idx)
                x = encoder_tokens[list_idx][:, :, patch_start_idx:]
            
            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx].contiguous()
            
            x = x.view(B * S, -1, x.shape[-1])

            x = self.norm(x)
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            
            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out, pts, H, W)

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H) 
        #removed as the post-fusion pos_embed injection as pts provides 3d positional encoding

        out = self.scratch.output_conv2(out)
        out = out.view(B, S, *out.shape[1:])
        return out