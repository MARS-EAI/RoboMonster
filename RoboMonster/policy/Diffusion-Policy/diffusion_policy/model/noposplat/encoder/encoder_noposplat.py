from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
# from ...dataset.shims.normalize_shim import apply_normalize_shim
# from ...dataset.types import BatchedExample, DataShim
# from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder


inf = float("inf")


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderNoPoSplatCfg:
    name: Literal["noposplat", "noposplat_multi"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    coor_type: Union['unify', 'self'] = 'unify'


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderNoPoSplat(Encoder[EncoderNoPoSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNoPoSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gs_params_head_type = cfg.gs_params_head_type

        self.set_center_head(
            output_mode="pts3d",
            head_type="dpt",
            landscape_only=True,
            depth_mode=("exp", -inf, inf),
            conf_mode=None,
        )
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

        self.coor_type = getattr(cfg, 'coor_type', 'unify')
    
    def wrapper_yes(self, head_id, decout, true_shape, ray_embedding=None):
        head = self.downstream_head1 if head_id == 1 else self.downstream_head2
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W), ray_embedding=ray_embedding)
        if is_portrait.all():
            return transposed(head(decout, (W, H), ray_embedding=ray_embedding))

        # batch is a mix of both portraint & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result = head(selout(is_landscape), (H, W), ray_embedding=ray_embedding)
        p_result = transposed(head(selout(is_portrait),  (W, H), ray_embedding=ray_embedding))

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

        # magic wrapper
        # self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        # self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def set_gs_params_head(self, cfg, head_type):
        if head_type == "linear":
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size**2 * self.raw_gs_dim,
                ),
            )

            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)
        elif head_type == "dpt":
            self.gaussian_param_head = head_factory(
                head_type, "gs_params", self.backbone, has_conf=False, out_nchan=self.raw_gs_dim
            )  # for view1 3DGS
            self.gaussian_param_head2 = head_factory(
                head_type, "gs_params", self.backbone, has_conf=False, out_nchan=self.raw_gs_dim
            )  # for view2 3DGS

        elif head_type == "dpt_gs":
            self.gaussian_param_head = head_factory(
                head_type, "gs_params", self.backbone, has_conf=False, out_nchan=self.raw_gs_dim
            )
            self.gaussian_param_head2 = head_factory(
                head_type, "gs_params", self.backbone, has_conf=False, out_nchan=self.raw_gs_dim
            )
        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        return self.wrapper_yes(head_num, decout, img_shape, ray_embedding=ray_embedding)

    def _forward(
        self,
        context: dict,
        global_step: int = 0,
    ) -> Gaussians:
        with torch.no_grad():      
            b, v, _, h, w = context["image"].shape
            intrinsics = torch.tensor([
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.0, 0.0, 1.0]
            ], device=context["image"].device)
            intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)           # [1, 1, 3, 3]
            intrinsics = intrinsics.expand(b, v, 3, 3)                  # [b, v, 3, 3]
            context["intrinsics"] = intrinsics

            # import pdb; pdb.set_trace()
            # Encode the context images.
            dec1, dec2, shape1, shape2, view1, view2 = self.backbone(context, return_views=True)
            with torch.amp.autocast("cuda", enabled=False):
                res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
                res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

                # for the 3DGS heads
                if self.gs_params_head_type == "linear":
                    GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
                    GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
                elif self.gs_params_head_type == "dpt":
                    GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
                    GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                    GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
                    GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
                elif self.gs_params_head_type == "dpt_gs":
                    GS_res1 = self.gaussian_param_head(
                        [tok.float() for tok in dec1],
                        res1["pts3d"].permute(0, 3, 1, 2),
                        view1["img"][:, :3],
                        shape1[0].cpu().tolist(),
                    )
                    GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                    GS_res2 = self.gaussian_param_head2(
                        [tok.float() for tok in dec2],
                        res2["pts3d"].permute(0, 3, 1, 2),
                        view2["img"][:, :3],
                        shape2[0].cpu().tolist(),
                    )
                    GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")

            pts3d1 = res1["pts3d"]
            pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
            pts3d2 = res2["pts3d"]
            pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
            pts_all = torch.stack((pts3d1, pts3d2), dim=1)
            pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

            depths = pts_all[..., -1].unsqueeze(-1)

            gaussians = torch.stack([GS_res1, GS_res2], dim=1)
            gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
            densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

            # Convert the features and depths into Gaussians.
            if self.pose_free:
                gaussians = self.gaussian_adapter.forward(
                    pts_all.unsqueeze(-2),
                    depths,
                    self.map_pdf_to_opacity(densities, global_step),
                    rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                )

            B, V, R, S, P, _ = gaussians.means.shape
            assert R == h * w, f"The number of rays should be equal to the number of pixels. But got R={R}, H={h}, W={w}."
            gaussians_final = Gaussians(
                rearrange(gaussians.means, "b v (h w) srf spp xyz -> b (v srf spp) h w xyz", h=h, w=w),
                rearrange(gaussians.covariances, "b v (h w) srf spp i j -> b (v srf spp) h w i j", h=h, w=w),
                rearrange(gaussians.harmonics, "b v (h w) srf spp c d_sh -> b (v srf spp) h w c d_sh", h=h, w=w),
                rearrange(gaussians.opacities, "b v (h w) srf spp -> b (v srf spp) h w", h=h, w=w),
            )

            B, N, H, W = gaussians_final.means.shape[:4]
            # means: 3D → C = 3
            means = gaussians_final.means.permute(0, 1, 4, 2, 3)  # [B, N, 3, H, W]

            # covariances: 3×3 matrix → flatten to 9D
            covs = gaussians_final.covariances.reshape(B, N, 9, H, W)

            # harmonics: [B, N, H, W, C, D_SH] → flatten to [B, N, C*D_SH, H, W]
            harmonics = gaussians_final.harmonics.reshape(B, N, -1, H, W).contiguous()

            # opacities: [B, N, H, W] → add channel dim → [B, N, 1, H, W]
            opacities = gaussians_final.opacities.unsqueeze(2)
            # gauss_feat = torch.cat([means, covs, harmonics, opacities], dim=2)  # [B, N, C', H, W]
            gauss_feat = torch.cat([means, covs, opacities], dim=2)  # [B, N, C', H, W]

        return gauss_feat

    def forward(
        self,
        context: dict,
        global_step: int = 0,
    ) -> tuple[Gaussians, Tensor]:

        with torch.no_grad():
            B, V, C, H, W = context["image"].shape
            if self.coor_type == 'unify':
                if V == 2:
                    gau_feat = self._forward(context, global_step)
                else:
                    image = context["image"]
                    # 取出第0帧并扩展，形状变为 [B, V-1, 3, H, W]
                    first = image[:, 0:1, :, :, :].expand(-1, V-1, -1, -1, -1)

                    # 取出第1到第V-1帧，形状 [B, V-1, 3, H, W]
                    others = image[:, 1:, :, :, :]

                    # 拼接第0帧与后面的每一帧，变为 [B, V-1, 2, 3, H, W]
                    pair = torch.stack([first, others], dim=2)
                    pair = pair.reshape(B*(V-1), 2, 3, H, W)

                    gau_feat = self._forward({'image': pair}, global_step)

                    gau_feat = gau_feat.reshape(B, (V-1), 2, 13, H, W)

                    gau_feat_view0 = gau_feat[:, :, 0].mean(dim=1, keepdim=True)
                    gau_feat_views = gau_feat[:, :, 1]
                    gau_feat = torch.cat([gau_feat_view0, gau_feat_views], dim=1)

            elif self.coor_type == 'self':
                image = context["image"]    # [B, V, 3, H, W]
                pair = torch.stack([image, torch.cat([image[:, 1:], image[:, 0:1]], dim=1)], dim=2)     # [B, V, 2, 3, H, W]
                pair = pair.reshape(B*V, 2, 3, H, W)

                gau_feat = self._forward({'image': pair}, global_step)
                gau_feat = gau_feat.reshape(B, V, 2, 13, H, W)[:, :, 0]

        return gau_feat