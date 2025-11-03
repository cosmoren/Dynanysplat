from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, Optional

from jaxtyping import Float
from torch import Tensor, nn

from ..types import Gaussians

DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]


@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"] # same as static if no dynamic gaussians
    depth: Float[Tensor, "batch view height width"] | None
    alpha: Float[Tensor, "batch view height width"] | None
    static_color: Float[Tensor, "batch view 3 height width"]
    static_depth: Float[Tensor, "batch view height width"] | None
    static_alpha: Float[Tensor, "batch view height width"] | None
    dynamic_color: Float[Tensor, "batch view 3 height width"] # zero_like if not dynamic gaussians
    dynamic_depth: Float[Tensor, "batch view height width"] | None
    dynamic_alpha: Float[Tensor, "batch view height width"] | None
    global_depth: Float[Tensor, "batch view height width"] | None
    lod_rendering: dict | None

T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg
    
    @abstractmethod
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        pass
