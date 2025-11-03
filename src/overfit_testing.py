"""
Overfitting Script - Load Images from Folder
This script loads a sequence of images from a folder and overfits your model to it.
Handles pretrained weight loading with partial loading for modified heads.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import glob

# Add your project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model components
from src.model.model import AnySplat
from src.loss.loss_mse import LossMse, LossMseCfg, LossMseCfgWrapper
from src.dataset.shims.normalize_shim import normalize_image, inverse_normalize_image

#config loading
import hydra
from omegaconf import DictConfig,OmegaConf
from src.config import load_typed_root_config
from src.model.model import get_model
from src.global_cfg import set_cfg
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# from src.dataset.data_module import DataModule
# from src.loss import get_losses
# from src.misc.LocalLogger import LocalLogger
# from src.misc.step_tracker import StepTracker
# from src.misc.wandb_tools import update_checkpoint_path
# from src.model.decoder import get_decoder
# from src.model.encoder import get_encoder
# from src.model.model_wrapper import ModelWrapper

def inspect_pretrained_weights(repo_id: str = "lhjiang/anysplat"):
    """
    Inspect the structure of pretrained weights
    
    Args:
        repo_id: HuggingFace model repo ID
    """
    from huggingface_hub import hf_hub_download
    import safetensors.torch
    
    print(f"\n{'='*60}")
    print(f"Inspecting pretrained weights from: {repo_id}")
    print(f"{'='*60}\n")
    
    try:
        # Try to download safetensors first (preferred format)
        cache_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors"
        )
        print(f"Cache location: {cache_path}\n")
        
        # Load weights
        weights = safetensors.torch.load_file(cache_path)
        
    except Exception as e:
        print(f"SafeTensors not found, trying pytorch_model.bin...")
        try:
            cache_path = hf_hub_download(
                repo_id=repo_id,
                filename="pytorch_model.bin"
            )
            print(f"Cache location: {cache_path}\n")
            weights = torch.load(cache_path, map_location="cpu")
        except Exception as e2:
            print(f"Error loading weights: {e2}")
            return None
    
    print("Weight structure:")
    print("-" * 60)
    
    # Group by module
    modules = {}
    for key, value in weights.items():
        module_name = key.split('.')[0] if '.' in key else key
        if module_name not in modules:
            modules[module_name] = []
        modules[module_name].append((key, value.shape))
    
    # Print organized
    for module_name in sorted(modules.keys()):
        print(f"\n{module_name}:")
        for key, shape in modules[module_name]:
            print(f"  {key}: {shape}")
    
    # Find gaussian head layers
    print(f"\n{'='*60}")
    print("Gaussian Parameter Head Layers:")
    print("-" * 60)
    for key, value in weights.items():
        if "gaussian_param_head" in key or "output_conv" in key or "scratch" in key:
            print(f"  {key}: {value.shape}")
    
    print(f"\n{'='*60}\n")
    
    return weights


def load_pretrained_with_modified_head(
    model: nn.Module,
    weight_path: str = None,
    repo_id: str = "lhjiang/anysplat",
    strict: bool = False,
    verbose: bool = True
) -> dict:
    """
    Load pretrained weights with partial channel initialization for modified layers
    
    Args:
        model: Your model instance
        weight_path: Local path to weight file (if None, downloads from repo_id)
        repo_id: HuggingFace repo ID (used if weight_path is None)
        strict: If True, all keys must match (will error on mismatch)
        verbose: Print loading details
    
    Returns:
        Dictionary with loading info (missing_keys, unexpected_keys, etc.)
    """
    import safetensors.torch
    
    print(f"\n{'='*60}")
    if weight_path:
        print(f"Loading pretrained weights from local path: {weight_path}")
    else:
        print(f"Loading pretrained weights from HuggingFace: {repo_id}")
    print(f"{'='*60}\n")
    
    # Load weights
    if weight_path:
        # Load from local path
        if weight_path.endswith('.safetensors'):
            pretrained_dict = safetensors.torch.load_file(weight_path)
        else:
            pretrained_dict = torch.load(weight_path, map_location="cpu")
    else:
        # Load from HuggingFace hub
        from huggingface_hub import hf_hub_download
        try:
            cache_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
            pretrained_dict = safetensors.torch.load_file(cache_path)
        except:
            print("SafeTensors not found, trying pytorch_model.bin...")
            cache_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            pretrained_dict = torch.load(cache_path, map_location="cpu")
    
    # Get model's current state dict
    model_dict = model.state_dict()
    
    # Track loading statistics
    matched_dict = {}
    mismatched_keys = []
    partial_loaded_keys = []
    
    for key, pretrained_value in pretrained_dict.items():
        if key not in model_dict:
            if verbose:
                print(f"Skipping key not in model: {key}")
            continue
        
        model_value = model_dict[key]
        
        # Case 1: Shapes match exactly - copy everything
        if pretrained_value.shape == model_value.shape:
            matched_dict[key] = pretrained_value
        
        # Case 2: Partial loading for mismatched output channels
        # This handles the case where we added extra output channels
        elif (len(pretrained_value.shape) >= 1 and 
              len(model_value.shape) == len(pretrained_value.shape)):
            
            # For Conv2d weights: (out_ch, in_ch, h, w)
            # For Conv2d bias: (out_ch,)
            # Check if only the first dimension (output channels) differs
            if (pretrained_value.shape[1:] == model_value.shape[1:] and 
                pretrained_value.shape[0] < model_value.shape[0]):
                
                # Partial copy - initialize shared channels from pretrained
                num_shared = pretrained_value.shape[0]
                new_tensor = model_value.clone()
                
                with torch.no_grad():
                    if len(pretrained_value.shape) == 4:  # Conv2d weight
                        new_tensor[:num_shared] = pretrained_value
                    elif len(pretrained_value.shape) == 1:  # Bias
                        new_tensor[:num_shared] = pretrained_value
                    else:  # Other tensors (linear, etc.)
                        new_tensor[:num_shared] = pretrained_value
                
                matched_dict[key] = new_tensor
                partial_loaded_keys.append(
                    f"{key}: Copied {num_shared}/{model_value.shape[0]} channels"
                )
                
                if verbose:
                    print(f"\n✓ Partial loading: {key}")
                    print(f"  Pretrained: {pretrained_value.shape}")
                    print(f"  Your model: {model_value.shape}")
                    print(f"  → Copied first {num_shared} channels")
                    print(f"  → Remaining {model_value.shape[0] - num_shared} channels keep random initialization")
            else:
                # Shape mismatch but not the pattern we handle
                mismatched_keys.append(
                    f"{key}: pretrained {pretrained_value.shape} vs model {model_value.shape}"
                )
        else:
            mismatched_keys.append(
                f"{key}: pretrained {pretrained_value.shape} vs model {model_value.shape}"
            )
    
    # Load the weights (both fully matched and partially loaded)
    model.load_state_dict(matched_dict, strict=False)
    
    # Report
    missing_keys = set(model_dict.keys()) - set(matched_dict.keys())
    fully_loaded = len(matched_dict) - len(partial_loaded_keys)
    
    print(f"\n{'='*60}")
    print(f"Loading Summary:")
    print(f"  ✓ Fully loaded: {fully_loaded} layers")
    print(f"  ⚠ Partially loaded: {len(partial_loaded_keys)} layers")
    print(f"  ⚠ Mismatched (skipped): {len(mismatched_keys)} layers")
    print(f"  ⚠ Missing in pretrained: {len(missing_keys)} layers")
    print(f"{'='*60}")

    # ========== HELPER FUNCTION: Truncate keys to depth 3 ==========
    def truncate_key_to_depth(key: str, max_depth: int = 3) -> str:
        """
        Truncate a key to a maximum depth
        
        Example:
            "encoder.blocks.0.attn.qkv.weight" (depth 5)
            → "encoder.blocks.0" (depth 3)
        """
        parts = key.split('.')
        if len(parts) <= max_depth:
            return key
        return '.'.join(parts[:max_depth])
    
    def get_missing_keys_grouped(missing_keys: set, max_depth: int = 3) -> dict:
        """
        Group missing keys by their truncated prefix (depth 3)
        
        Returns:
            Dictionary mapping truncated_key → list of full keys
        """
        grouped = {}
        for key in missing_keys:
            truncated = truncate_key_to_depth(key, max_depth)
            if truncated not in grouped:
                grouped[truncated] = []
            grouped[truncated].append(key)
        return grouped
    # ========== END HELPER FUNCTION ==========
    
    if partial_loaded_keys and verbose:
        print(f"\nPartially initialized layers (copied shared channels):")
        for info in partial_loaded_keys:
            print(f"  - {info}")
    
    if mismatched_keys and verbose:
        print(f"\nMismatched layers (will use random initialization):")
        for key in mismatched_keys[:10]:  # Show first 10 only
            print(f"  - {key}")
        if len(mismatched_keys) > 10:
            print(f"  ... and {len(mismatched_keys) - 10} more")
    
    # ========== PRINT MISSING KEYS WITH DEPTH 3 ==========
    if missing_keys and verbose:
        print(f"\nMissing in pretrained (will use random initialization):")
        
        # Group by depth-3 prefix
        grouped_missing = get_missing_keys_grouped(missing_keys, max_depth=3)
        
        # Sort by prefix for readability
        for truncated_key in sorted(grouped_missing.keys()):
            full_keys = grouped_missing[truncated_key]
            
            if len(full_keys) == 1:
                # Single key - show full key
                print(f"  - {full_keys[0]}")
            else:
                # Multiple keys with same prefix - show grouped
                print(f"  - {truncated_key}.* ({len(full_keys)} keys)")
                
                # Optionally show a few examples
                for example_key in full_keys[:3]:
                    print(f"      └─ {example_key}")
                if len(full_keys) > 3:
                    print(f"      └─ ... and {len(full_keys) - 3} more")
        
        print(f"\n  Total missing keys: {len(missing_keys)}")
        print(f"  Unique prefixes (depth 3): {len(grouped_missing)}")
    # ========== END MISSING KEYS PRINTING ==========
    
    print(f"\n{'='*60}\n")
    
    return {
        "loaded": len(matched_dict),
        "partial_loaded": partial_loaded_keys,
        "mismatched": mismatched_keys,
        "missing": list(missing_keys),
    }


def load_images_from_folder(
    folder_path: str,
    num_views: Optional[int] = None,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'],
    target_size: Optional[tuple] = None,
    sort_by_name: bool = True,
) -> torch.Tensor:
    """
    Load images from a folder
    
    Args:
        folder_path: Path to folder containing images
        num_views: Number of images to load (None = all images)
        image_extensions: Valid image extensions
        target_size: (H, W) to resize images to, or None to keep original
        sort_by_name: Sort images by filename
    
    Returns:
        images: Tensor of shape (1, V, 3, H, W)
    """
    folder_path = Path(folder_path)
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(str(folder_path / f"*{ext}")))
    
    if not image_files:
        raise ValueError(f"No images found in {folder_path}")
    
    # Sort if requested
    if sort_by_name:
        image_files = sorted(image_files)
    
    # Limit number of views
    if num_views is not None:
        image_files = image_files[:num_views]
    
    print(f"Loading {len(image_files)} images from {folder_path}")
    
    # Load images
    images = []
    for img_path in tqdm(image_files, desc="Loading images"):
        img = Image.open(img_path).convert('RGB')
        
        # Resize if needed
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        
        # Convert to tensor [0, 1]
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
        
        images.append(img_tensor)
    
    images = torch.stack(images).unsqueeze(0)  # (1, V, 3, H, W)
    
    print(f"Loaded images shape: {images.shape}")
    return images


def create_simple_poses(
    num_views: int,
    camera_distance: float = 2.0,
    fov: float = 60.0,
    image_size: tuple = (392, 518),
) -> tuple:
    """
    Create simple camera poses in a circle
    
    Args:
        num_views: Number of views
        camera_distance: Distance from origin
        fov: Field of view in degrees
        image_size: (H, W)
    
    Returns:
        extrinsics: (1, V, 4, 4)
        intrinsics: (1, V, 3, 3)
    """
    h, w = image_size
    
    # Create extrinsics (cameras in a circle)
    extrinsics = []
    angles = torch.linspace(0, 2 * np.pi, num_views + 1)[:-1]  # Exclude last point
    
    for angle in angles:
        # Camera position
        x = camera_distance * torch.cos(angle)
        z = camera_distance * torch.sin(angle)
        y = 0.0
        
        # Create camera-to-world matrix
        camera_pos = torch.tensor([x, y, z])
        look_at = torch.tensor([0.0, 0.0, 0.0])
        up = torch.tensor([0.0, 1.0, 0.0])
        
        # Z-axis: camera direction
        z_axis = (look_at - camera_pos)
        z_axis = z_axis / z_axis.norm()
        
        # X-axis: right
        x_axis = torch.cross(z_axis, up)
        x_axis = x_axis / x_axis.norm()
        
        # Y-axis: up (recompute for orthogonality)
        y_axis = torch.cross(x_axis, z_axis)
        
        # Create rotation matrix
        R = torch.stack([x_axis, y_axis, -z_axis], dim=1)  # (3, 3)
        
        # Create 4x4 matrix (camera-to-world)
        c2w = torch.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = camera_pos
        
        # Convert to world-to-camera (extrinsics)
        w2c = c2w.inverse()
        
        extrinsics.append(w2c)
    
    extrinsics = torch.stack(extrinsics).unsqueeze(0)  # (1, V, 4, 4)
    
    # Create intrinsics
    focal_length = (w / 2.0) / np.tan(np.radians(fov / 2.0))
    
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, num_views, 1, 1)
    intrinsics[:, :, 0, 0] = focal_length / w  # Normalized fx
    intrinsics[:, :, 1, 1] = focal_length / h  # Normalized fy
    intrinsics[:, :, 0, 2] = 0.5  # Normalized cx
    intrinsics[:, :, 1, 2] = 0.5  # Normalized cy
    
    print(f"Created simple poses: {num_views} views in a circle")
    
    return extrinsics, intrinsics


def create_batch_from_folder(
    folder_path: str,
    num_views: Optional[int] = None,
    target_size: tuple = (775, 1024),
    camera_distance: float = 2.0,
    fov: float = 60.0,
) -> dict:
    """
    Create a batch from a folder of images
    
    Args:
        folder_path: Path to image folder
        num_views: Number of views to load (None = all)
        target_size: (H, W) to resize images
        camera_distance: Camera distance for pose generation
        fov: Field of view in degrees
    
    Returns:
        batch: Dictionary containing images and poses
    """
    # Load images
    images = load_images_from_folder(
        folder_path,
        num_views=num_views,
        target_size=target_size,
    )
    
    b, v, c, h, w = images.shape
    
    # Create poses
    extrinsics, intrinsics = create_simple_poses(
        num_views=v,
        camera_distance=camera_distance,
        fov=fov,
        image_size=(h, w),
    )
    
    # Normalize images to [-1, 1]
    images = normalize_image(images, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    # Create valid mask
    valid_mask = torch.ones(1, v, h, w, dtype=torch.bool)
    
    # Create batch
    batch = {
        "context": {
            "image": images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "near": torch.ones(1, v) * 0.01,
            "far": torch.ones(1, v) * 1000.0,
            "index": torch.arange(v).unsqueeze(0),
            "valid_mask": valid_mask,
        },
        "scene": f"folder_{Path(folder_path).name}",
    }
    
    print(f"\nBatch created:")
    print(f"  Images: {images.shape}")
    print(f"  Extrinsics: {extrinsics.shape}")
    print(f"  Intrinsics: {intrinsics.shape}")
    
    return batch


class OverfitTrainer:
    """Trainer for overfitting on a single batch"""
    
    def __init__(
        self,
        model: AnySplat,
        device: str = "cuda",
        lr: float = 1e-8,
        backbone_lr_multiplier: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.backbone_lr_multiplier = backbone_lr_multiplier
        
        # Set up loss
        loss_cfg = LossMseCfgWrapper(
            mse=LossMseCfg(weight=1.0, conf=True, mask=False, alpha=False)
        )
        self.loss_fn = LossMse(loss_cfg)
        
        # Set up optimizer
        self.optimizer = self._setup_optimizer()
        
        # Tracking
        self.losses = []
        self.psnrs = []
        self.global_step = 0
        
    def _setup_optimizer(self):
        """
        Set up optimizer with learning rates:
        - gaussian_param_head: Normal learning rate (self.lr)
        - All other components: Zero learning rate (frozen)
        """
        dynamic_head_params = []
        frozen_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Only train dynamic_prob_head
            if "dynamic_prob_head" in name:
                dynamic_head_params.append(param)
            else:
                frozen_params.append(param)
        
        param_groups = [
            {"params": dynamic_head_params, "lr": self.lr},
            {"params": frozen_params, "lr": 0.0},  # Frozen (zero learning rate)
        ]
        
        print(f"\n{'='*60}")
        print(f"Optimizer Setup:")
        print(f"{'='*60}")
        print(f"  Dynamic Head params: {sum(p.numel() for p in dynamic_head_params):,} parameters")
        print(f"    Learning rate: {self.lr:.2e} ✓ TRAINABLE")
        print(f"\n  Other components: {sum(p.numel() for p in frozen_params):,} parameters")
        print(f"    Learning rate: 0.0 ✗ FROZEN")
        print(f"{'='*60}\n")
        
        optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self.lr, 
            weight_decay=0.05, 
            betas=(0.9, 0.95)
        )
        
        return optimizer
    
    def prepare_batch(self, batch: dict) -> dict:
        """Prepare batch for training"""
        # Move to device
        for key in ["context"]:
            for subkey in batch[key]:
                if isinstance(batch[key][subkey], torch.Tensor):
                    batch[key][subkey] = batch[key][subkey].to(self.device)
        
        # Add using_index
        b, v = batch["context"]["image"].shape[:2]
        batch["using_index"] = torch.arange(v, device=self.device)
        return batch
    
    def train_step(self, batch: dict) -> dict:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Get normalized images (range [-1, 1])
        images_normalized = batch["context"]["image"]
        
        # Denormalize for model input (range [0, 1])
        context_image = inverse_normalize_image(
            images_normalized,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )

        # print(f"context image for training:")
        # print(f"  Min:  {context_image.min():.6f}")
        # print(f"  Max:  {context_image.max():.6f}")
        # print(f"  Mean: {context_image.mean():.6f}")
        # print(f"  Std:  {context_image.std():.6f}")
        
        # Forward pass through model
        encoder_output, decoder_output = self.model(
            context_image, 
            self.global_step,
            visualization_dump=None,
        )
        static_depth = decoder_output.depth
        raw_depth = encoder_output.depth_dict.get("depth")[..., 0]
        
        # Compute loss
        loss = self.loss_fn(
            prediction=decoder_output,
            batch=batch,
            gaussians=encoder_output.gaussians,
            depth_dict=encoder_output.depth_dict,
            global_step=self.global_step,
            static_depth=static_depth,
            raw_depth=raw_depth,
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Track metrics
        self.losses.append(loss.item())
        self.global_step += 1
        
        # Compute PSNR
        with torch.no_grad():
            pred_img = decoder_output.color.permute(0, 1, 3, 4, 2)
            gt_img = context_image.permute(0, 1, 3, 4, 2)
            
            mse = ((pred_img - gt_img) ** 2).mean()
            psnr = -10 * torch.log10(mse + 1e-8)
            self.psnrs.append(psnr.item())
        
        return {
            "loss": loss.item(),
            "psnr": psnr.item(),
            "num_static_gaussians": encoder_output.infos.get("num_static_gaussians", 0),
            "num_dynamic_gaussians": encoder_output.infos.get("num_dynamic_gaussians", 0),
        }
    
    def train(
        self, 
        batch: dict, 
        num_steps: int = 1000,
        log_every: int =500,
        save_every: int = 500,
        save_dir: str = "./overfit_results"
    ):
        """Train on a single batch"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare batch
        batch = self.prepare_batch(batch)
        
        # Run prediction BEFORE training to see initial performance
        print("\nRunning initial prediction before training...")
        self.visualize_results(batch, f"{save_dir}/vis_init.png")

        print(f"\n{'='*60}")
        print(f"Starting overfitting for {num_steps} steps")
        print(f"{'='*60}")
        print(f"Batch shape: {batch['context']['image'].shape}")
        print(f"Device: {self.device}")
        print(f"Save directory: {save_dir}")
        print(f"{'='*60}\n")
        
        # Training loop
        pbar = tqdm(range(num_steps), desc="Training")
        for step in pbar:
            metrics = self.train_step(batch)
            
            # Logging
            if step % log_every == 0:
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.6f}",
                    "psnr": f"{metrics['psnr']:.2f}",
                    "static": metrics['num_static_gaussians'],
                    "dynamic": metrics['num_dynamic_gaussians'],
                })
            
            # Save checkpoint and visualizations
            if step % save_every == 0 and step > 0:
                # self.save_checkpoint(save_dir / f"checkpoint_step_{step}.pt")
                self.visualize_results(batch, save_dir / f"vis_step_{step}.png")
                self.plot_metrics(save_dir)
        
        # Final save
        # self.save_checkpoint(save_dir / "checkpoint_final.pt")
        self.visualize_results(batch, save_dir / "vis_final.png", all_frames=True)
        self.plot_metrics(save_dir)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"{'='*60}")
        print(f"Final loss: {metrics['loss']:.6f}")
        print(f"Final PSNR: {metrics['psnr']:.2f} dB")
        print(f"Results saved to: {save_dir}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "losses": self.losses,
            "psnrs": self.psnrs,
        }, path)
    
    @torch.no_grad()
    def visualize_results(self, batch: dict, save_path: Path, all_frames=False):
        """Visualize predictions vs ground truth"""
        self.model.eval()
        
        # Get predictions
        images_normalized = batch["context"]["image"]
        context_image = inverse_normalize_image(
            images_normalized,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
        
        encoder_output, decoder_output = self.model(
            context_image, 
            self.global_step,
            visualization_dump=None,
            near=batch["context"]["near"][0, 0].item(),
            far=batch["context"]["far"][0, 0].item(),
        )
        
        self.model.train()
        
        # Get images (first view only)
        pred_img = decoder_output.color[0, 0].permute(1, 2, 0).cpu().numpy()
        gt_img = context_image[0, 0].permute(1, 2, 0).cpu().numpy()
        static_frame = decoder_output.static_color[0, 0].permute(1, 2, 0).cpu().numpy() 
        dynamic_frame = decoder_output.dynamic_color[0, 0].permute(1, 2, 0).cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        
        axes[0].imshow(np.clip(gt_img, 0, 1))
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        
        # print(f"Prediction (before clipping):")
        # print(f"  Min:  {pred_img.min():.6f}")
        # print(f"  Max:  {pred_img.max():.6f}")
        # print(f"  Mean: {pred_img.mean():.6f}")
        # print(f"  Std:  {pred_img.std():.6f}")

        axes[1].imshow(np.clip(pred_img, 0, 1))
        axes[1].set_title("Prediction")
        axes[1].axis("off")
        
        axes[2].imshow(np.clip(static_frame, 0, 1))
        axes[2].set_title(f"Static")
        axes[2].axis("off")

        axes[3].imshow(np.clip(dynamic_frame, 0, 1))
        axes[3].set_title(f"Dynamic")
        axes[3].axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        if all_frames:
            # visualize the rest of the frames, storeing four frames images for each saved fig, comparing the GT and pred of the frame
            num_views = decoder_output.color.shape[1]
            frames_per_fig = 4
            num_figs = (num_views + frames_per_fig - 1) // frames_per_fig  # Ceiling division
            
            save_dir = save_path.parent
            save_stem = save_path.stem  # Filename without extension
            
            for fig_idx in range(num_figs):
                start_idx = fig_idx * frames_per_fig
                end_idx = min(start_idx + frames_per_fig, num_views)
                num_frames_in_fig = end_idx - start_idx

                # Create figure with 3 rows (GT, Pred, Static) and num_frames_in_fig columns
                fig, axes = plt.subplots(4, num_frames_in_fig, figsize=(5 * num_frames_in_fig, 12))
                
                # Handle case where num_frames_in_fig == 1
                if num_frames_in_fig == 1:
                    axes = axes.reshape(4, 1)
                
                for col_idx, view_idx in enumerate(range(start_idx, end_idx)):
                    # Get GT and prediction for this view
                    gt_frame = context_image[0, view_idx].permute(1, 2, 0).cpu().numpy()
                    pred_frame = decoder_output.color[0, view_idx].permute(1, 2, 0).cpu().numpy()
                    static_frame = decoder_output.static_color[0, view_idx].permute(1, 2, 0).cpu().numpy() 
                    dynamic_frame = decoder_output.dynamic_color[0, view_idx].permute(1, 2, 0).cpu().numpy()

                    # Row 0: Ground Truth
                    axes[0, col_idx].imshow(np.clip(gt_frame, 0, 1))
                    axes[0, col_idx].set_title(f"GT View {view_idx}")
                    axes[0, col_idx].axis("off")
                    
                    # Row 1: Prediction
                    axes[1, col_idx].imshow(np.clip(pred_frame, 0, 1))
                    axes[1, col_idx].set_title(f"Pred View {view_idx}")
                    axes[1, col_idx].axis("off")
                    
                    # Row 2: static background
                    axes[2, col_idx].imshow(np.clip(static_frame, 0, 1))
                    axes[2, col_idx].set_title(f"Static Frame View {view_idx}")
                    axes[2, col_idx].axis("off")

                    # Row 3: Dynamic Frame
                    axes[3, col_idx].imshow(np.clip(dynamic_frame, 0, 1))
                    axes[3, col_idx].set_title(f"Dynamic Frame View {view_idx}")
                    axes[3, col_idx].axis("off")

                    # # Add colorbar only to the last column
                    # if col_idx == num_frames_in_fig - 1:
                    #     plt.colorbar(im, ax=axes[2, col_idx])
                
                plt.tight_layout()
                
                # Save with unique filename
                all_frames_path = save_dir / f"{save_stem}_all_frames_{fig_idx}.png"
                plt.savefig(all_frames_path, dpi=150, bbox_inches="tight")
                plt.close()
                
                print(f"Saved all-frames visualization {fig_idx + 1}/{num_figs}: {all_frames_path}")
            
    def plot_metrics(self, save_dir: Path):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        axes[0].plot(self.losses)
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True)
        axes[0].set_yscale("log")
        
        # PSNR curve
        axes[1].plot(self.psnrs)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("PSNR (dB)")
        axes[1].set_title("PSNR")
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / "metrics.png", dpi=150, bbox_inches="tight")
        plt.close()
    

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def main(cfg_dict: DictConfig):
    """Main training script"""
    
    # ========== CONFIGURATION ==========
    # Original Omega conf, used for initializing model
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    model = get_model(cfg.model.encoder, cfg.model.decoder)
    # model = AnySplat.from_pretrained("lhjiang/anysplat")


    # Image folder path - CHANGE THIS!
    IMAGE_FOLDER = "/home/yuan/workspace/sq/datasets/av2_overfit"
    
    # Number of images to load (None = all images in folder)
    NUM_VIEWS = 9
    
    # Target image size (H, W) - images will be resized to this
    TARGET_SIZE = (196, 266)  # Or use (196, 259) for faster training
    
    # Camera configuration for pose generation
    CAMERA_DISTANCE = 2.0  # Distance from scene center
    CAMERA_FOV = 60.0  # Field of view in degrees
    
    # Training configuration
    NUM_STEPS = 2000
    LEARNING_RATE = 1e-4
    SAVE_DIR = "/home/yuan/workspace/sq/AnySplat/overfit_results/additional_head_training"
    
    # Pretrained weights configuration
    PRETRAINED_WEIGHT_PATH = "/home/yuan/.cache/huggingface/hub/models--lhjiang--anysplat/snapshots/d2e8c343672646041ad4ea518184968f94362f01/model.safetensors"
    USE_LOCAL_WEIGHTS = False  # Set to False to download from HuggingFace
    PRETRAINED_REPO = "lhjiang/anysplat"  # Used if USE_LOCAL_WEIGHTS=False
    
    INSPECT_WEIGHTS = False  # Set to True to inspect weight structure
    
    # ========== END CONFIGURATION ==========
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Inspect pretrained weights (optional)
    if INSPECT_WEIGHTS:
        if USE_LOCAL_WEIGHTS:
            print(f"\nInspecting local weights from: {PRETRAINED_WEIGHT_PATH}")
            import safetensors.torch
            weights = safetensors.torch.load_file(PRETRAINED_WEIGHT_PATH)
            print("\nGaussian Parameter Head Layers:")
            for key, value in weights.items():
                if "gaussian_param_head" in key:
                    print(f"  {key}: {value.shape}")
        else:
            inspect_pretrained_weights(PRETRAINED_REPO)
        print("\nWeights inspected! Set INSPECT_WEIGHTS=False to continue training.\n")
        return
    
    # Load images from folder
    print("\nLoading images from folder...")
    batch = create_batch_from_folder(
        folder_path=IMAGE_FOLDER,
        num_views=NUM_VIEWS,
        target_size=TARGET_SIZE,
        camera_distance=CAMERA_DISTANCE,
        fov=CAMERA_FOV,
    )
    
    
    # Load pretrained weights with partial channel initialization
    print("\nLoading pretrained weights...")
    load_info = load_pretrained_with_modified_head(
        model,
        weight_path=PRETRAINED_WEIGHT_PATH if USE_LOCAL_WEIGHTS else None,
        repo_id=PRETRAINED_REPO if not USE_LOCAL_WEIGHTS else None,
        strict=False,
        verbose=True
    )
    
    print(f"\n✓ Loaded {load_info['loaded']} parameters")
    if load_info['partial_loaded']:
        print(f"✓ Partially loaded {len(load_info['partial_loaded'])} layers with shared channel initialization")
    
    print("\nModel created and weights loaded successfully!")
    
    # Create trainer (Note: backbone_lr_multiplier is ignored now - only gaussian_param_head trains)
    print("\nSetting up trainer...")
    trainer = OverfitTrainer(
        model=model,
        device=device,
        lr=LEARNING_RATE,
        backbone_lr_multiplier=0.0,  # Not used anymore - all non-gaussian_param_head frozen
    )
    
    
    # Train
    print("\nStarting training...")
    trainer.train(
        batch=batch,
        num_steps=NUM_STEPS,
        log_every=100,
        save_every=500,
        save_dir=SAVE_DIR,
    )
    
    print(f"\nTraining complete!")
    print(f"\nResults saved in: {SAVE_DIR}")


if __name__ == "__main__":
    main()