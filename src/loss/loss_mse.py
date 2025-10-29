from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn.functional as F
from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    conf: bool = False
    mask: bool = False
    alpha: bool = False


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Get alpha and valid mask from inputs
        alpha = prediction.alpha
        # valid_mask = torch.ones_like(alpha, device=alpha.device).bool()
        valid_mask = batch['context']['valid_mask']

        # # only for objaverse
        # if batch['context']['valid_mask'].sum() > 0:
        #     valid_mask = batch['context']['valid_mask']

        # Determine which mask to use based on config
        if self.cfg.mask:
            mask = valid_mask
        elif self.cfg.alpha:
            mask = alpha  
        elif self.cfg.conf:
            mask = depth_dict['conf_valid_mask']
        else:
            mask = torch.ones_like(alpha, device=alpha.device).bool()

        # Rearrange and mask predicted and ground truth images
        pred_img = prediction.color.permute(0, 1, 3, 4, 2)[mask] 
        gt_img = ((batch["context"]["image"][:, batch["using_index"]] + 1) / 2).permute(0, 1, 3, 4, 2)[mask]

        delta = pred_img - gt_img

        # static version, use the static gaussians' rendering, but with the same mask and gt_img
        static_img = prediction.static_color.permute(0, 1, 3, 4, 2)[mask]
        static_delta = static_img - gt_img
        static_weight = 0.5
        
        # pushing dynamic and static to be away from each other
        dynamic_img = prediction.dynamic_color.permute(0, 1, 3, 4, 2)[mask]
        alpha_weight= prediction.dynamic_alpha[mask] * prediction.static_alpha[mask]
        cos_sim = F.cosine_similarity(
            dynamic_img, 
            static_img, 
            dim=-1
        ).abs()
        similarity_loss = (cos_sim * alpha_weight).mean() #(cos_sim * weights).sum() # / (weights.sum() + 1e-6) might need this term for stablizing gradient
        similarity_weight = 0.01

        #print(f"debug: delta-mean: {delta.mean().item()}, static-delta-mean: {static_delta.mean().item()}, dynamic-static-sim: {(similarity_loss * similarity_weight).item()}")


        combined_loss = torch.nan_to_num((delta**2).mean(), nan=0.0, posinf=0.0, neginf=0.0)
        static_loss = torch.nan_to_num((static_delta**2 * static_weight).mean(), nan=0.0, posinf=0.0, neginf=0.0)
        dynamic_sim_loss = torch.nan_to_num(similarity_loss * similarity_weight, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"debug: mse-loss: {combined_loss.item()}, static-mse-loss: {static_loss.item()}, dynamic-static-sim-loss: {dynamic_sim_loss.item()}")
        return self.cfg.weight * (combined_loss + static_loss + dynamic_sim_loss)