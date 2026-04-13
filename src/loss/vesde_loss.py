import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.ema import ExponentialMovingAverage as EMA

# 确保能引用到 diffusionmbir 包
# 如果你的目录结构不同，请调整这里的 path append
sys.path.append("./diffusionmbir")
from ..models import utils as mutils
from ..utils import sde_lib


class VESDEGuidance(nn.Module):
    def __init__(
        self,
        config_path,
        ckpt_path,
        annealing=False,
        device="cuda",
        use_paas=False,
        paas_k=1,
        paas_rho=0.1,
    ):
        super().__init__()
        self.annealing = annealing
        self.device = device
        self.use_paas = bool(use_paas)
        self.paas_k = max(int(paas_k), 1)
        self.paas_rho = float(paas_rho)

        # 1. 动态加载 Config (.py 文件)
        self.config = self._load_config_from_path(config_path)

        # 2. 实例化 SDE (必须存在，供计算 score 和 marginal_prob 使用)
        self.sde = sde_lib.VESDE(
            sigma_min=self.config.model.sigma_min,
            sigma_max=self.config.model.sigma_max,
            N=self.config.model.num_scales,
        )
        # 3. 创建空模型
        self.model = mutils.create_model(self.config)
        self.model.to(device)
        self.model.eval()

        # 4. 加载 Checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # 5. [关键] "借用" EMA 提取最佳权重，用完即弃
        if "ema" in checkpoint:
            # 初始化一个临时的 EMA 工具
            ema_helper = EMA(self.model.parameters(), decay=self.config.model.ema_rate)
            # 载入 EMA 状态
            ema_helper.load_state_dict(checkpoint["ema"])
            # 核心动作：把 EMA 的好权重覆盖到 self.model 上
            ema_helper.copy_to(self.model.parameters())
            print("Loaded EMA weights (Best Quality).")
            # 此时 ema_helper 局部变量会被自动回收，不占用显存
        else:
            # 如果没 EMA，只能用普通权重
            self.model.load_state_dict(checkpoint["model"])

        # 6. 彻底冻结模型，只做前向计算
        for p in self.model.parameters():
            p.requires_grad = False

        self.eps = 1e-5

    def _load_config_from_path(self, path):
        """Helper to load config from a python file path."""
        spec = importlib.util.spec_from_file_location("config", path)
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        return cfg_module.get_config()

    def train_step(self, x0, step_ratio=None):
        """
        计算 SDS Gradient 的核心步骤
        Args:
            x0: [B, C, H, W] - 来自 NAF 的干净切片预测 (pred_slices)
            step_ratio: (Optional) 用于控制时间步采样的范围
        Returns:
            loss_sds: Scalar Tensor (Surrogate Loss), backward 后会产生 SDS 梯度
        """
        batch_size = x0.shape[0]

        # 1. 采样时间步 t
        # VE-SDE 的时间范围通常是 [0, 1] (continuous)
        # 为了数值稳定性，通常避免 t=0，使用极小值 eps
        eps = 1e-5
        if self.annealing:
            t = self.sample_t_annealing(batch_size, step_ratio)
        else:
            t = torch.rand(batch_size, device=self.device) * (self.sde.T - eps) + eps

        # 2. 前向加噪 (Forward SDE)
        # x_t = x_0 + sigma_t * z
        z = torch.randn_like(x0)
        mean, std = self.sde.marginal_prob(x0, t)
        # mean 通常就是 x0 (对于 VE-SDE), std 是 sigma_t
        sigma_t = std[:, None, None, None]
        x_t = mean + sigma_t * z

        # 3. 预测 Score (Denoise)
        with torch.no_grad():
            # 获取 score function (内部会自动处理 scale_by_sigma 等逻辑)
            # continuous=True 表示我们使用的是连续时间 SDE
            score_fn = mutils.get_score_fn(
                self.sde, self.model, train=False, continuous=True
            )

            # 计算 score: s_theta(x_t, t)
            # 这里的 score_pred 近似于 \nabla log p_t(x_t)
            if self.use_paas and self.paas_k > 1:
                score_pred = self._predict_score_paas(score_fn, x_t, t, sigma_t)
            else:
                score_pred = score_fn(x_t, t)

        # 4. 计算 SDS 梯度 (Residual)
        # 理论推导:
        # Score Target (Real) = -z / sigma_t
        # Grad Direction ~ w(t) * (Score Pred - Score Target)
        #                = w(t) * (Score Pred + z / sigma_t)
        #
        # 权重选择 (Likelihood Weighting): w(t) = sigma_t^2
        # Grad = sigma_t^2 * (Score Pred + z / sigma_t)
        #      = sigma_t * (Score Pred * sigma_t + z)
        # 注意: (Score Pred * sigma_t) 量级约为 -z (Unit Variance)
        # 所以 (Score Pred * sigma_t + z) 近似于 (Noise Pred - Noise Real)

        # 使用 Noise Residual 形式的梯度 (更加稳定):
        grad = score_pred * sigma_t + z

        # 5. 构建 Surrogate Loss
        # 我们希望 x0 的梯度方向是 grad
        # L_sds = 0.5 * || x0 - (x0 - grad).detach() ||^2
        # 对 L_sds 求导 => dL/dx0 = (x0 - (x0 - grad)) = grad

        target = (x0 + grad).detach()
        loss_sds = 0.5 * F.mse_loss(x0, target, reduction="mean")

        return loss_sds
    
    def train_step_with_Fidelity(
        self,
        x0_vol,
        gt_projs,
        rays,
        bound_box,
        n_samples=512,
        rays_chunk=4096,
        fidelity_weight=0.05,
        step_ratio=None,
        slice_chunk=8,
    ):
        """
        SDS + DPS 保真度项的训练步骤

        Args:
            x0_vol: [1, 1, D, H, W] 完整体积（from NAF）
            gt_projs: [N_rays] 或 [N_angles, H, W] GT 投影
            rays: [N_rays, 8] (rays_o, rays_d, near, far)
            bound_box: [3] 物理边界 (x,y,z)
            n_samples: int 每条光线的采样点数
            rays_chunk: int 每次投影处理的 rays 数量
            fidelity_weight: float λ 梯度修正权重
            step_ratio: float 用于 annealing 时间步调度
            slice_chunk: int 切片批大小，避免 OOM

        Returns:
            loss_total: scalar loss
        """
        if x0_vol.dim() != 5:
            raise ValueError(f"x0_vol must be [1,1,D,H,W], got {x0_vol.shape}")
        if rays.dim() != 2 or rays.shape[-1] != 8:
            raise ValueError(f"rays must be [N_rays,8], got {rays.shape}")
        if gt_projs.numel() != rays.shape[0]:
            raise ValueError(
                f"gt_projs size {gt_projs.numel()} must match N_rays {rays.shape[0]}"
            )
        if not torch.is_tensor(bound_box) or bound_box.numel() != 3:
            raise ValueError("bound_box must be a tensor with 3 elements [x,y,z]")
        if slice_chunk < 1:
            raise ValueError("slice_chunk must be >= 1")
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        if rays_chunk < 1:
            raise ValueError("rays_chunk must be >= 1")

        _, _, D, H, W = x0_vol.shape

        # ========== 步骤 1-4: 分块计算 score 与 x0_hat ==========
        eps = 1e-5
        t_full = (
            self.sample_t_annealing(D, step_ratio)
            if self.annealing
            else torch.rand(D, device=self.device) * (self.sde.T - eps) + eps
        )

        x0_slices = x0_vol.squeeze(0).squeeze(0).unsqueeze(1)  # [D, 1, H, W]
        z = torch.randn_like(x0_slices)
        mean, std = self.sde.marginal_prob(x0_slices, t_full)
        sigma_t = std[:, None, None, None]
        x_t = mean + sigma_t * z

        x_t.requires_grad_(True)

        grad_sds_list = []
        x0_hat_list = []
        sigma_t_list = []

        # with torch.no_grad():
        score_fn = mutils.get_score_fn(
            self.sde, self.model, train=False, continuous=True
        )

        for i in range(0, D, slice_chunk):
            j = min(i + slice_chunk, D)
            # x0_chunk = x0_slices[i:j].unsqueeze(1)  # [B, 1, H, W]
            x_t_chunk = x_t[i:j]
            t_chunk = t_full[i:j]

            # z = torch.randn_like(x0_chunk)
            # mean, std = self.sde.marginal_prob(x0_chunk, t_chunk)
            # sigma_t = std[:, None, None, None]
            # x_t = mean + sigma_t * z

            if self.use_paas and self.paas_k > 1:
                score_pred = self._predict_score_paas(
                    score_fn, x_t_chunk, t_chunk, sigma_t
                )
            else:
                score_pred = score_fn(x_t_chunk, t_chunk)

            x0_hat = x_t_chunk + (sigma_t**2) * score_pred

            grad_sds = score_pred * sigma_t + z
            grad_sds_list.append(grad_sds)
            x0_hat_list.append(x0_hat)
            sigma_t_list.append(sigma_t)

        grad_sds = torch.cat(grad_sds_list, dim=0).unsqueeze(
            0
        )  # [1, D, 1, H, W] -> [1, D, 1, H, W]
        grad_sds = grad_sds.permute(0, 2, 1, 3, 4)  # [1, 1, D, H, W]

        x0_hat = torch.cat(x0_hat_list, dim=0).unsqueeze(0)
        x0_hat = x0_hat.permute(0, 2, 1, 3, 4)  # [1, 1, D, H, W]

        sigma_t_full = torch.cat(sigma_t_list, dim=0).unsqueeze(0)
        sigma_t_full = sigma_t_full.permute(0, 2, 1, 3, 4)  # [1, 1, D, H, W]

        # ========== 步骤 5a: Rays + grid_sample 正向投影 ==========
        # x0_hat_proj = x0_hat.clone().detach()

        proj_pred = self._project_volume_with_rays(
            x0_hat, rays, bound_box, n_samples, rays_chunk
        )

        # ========== 步骤 5b: 计算投影残差梯度 ==========
        loss_fidelity_proj = 0.5 * F.mse_loss(proj_pred, gt_projs, reduction="sum")

        # 计算梯度 ∇_{x0_hat} ||y - A(x0_hat)||^2
        grad_fidelity = torch.autograd.grad(
            loss_fidelity_proj,
            x_t,
            create_graph=False,
            retain_graph=False,
        )[
            0
        ]  # [1, 1, D, H, W]

        # ========== 步骤 5c: 梯度修正 ==========
        grad_corrected = (
            grad_sds - fidelity_weight * sigma_t_full * grad_fidelity
        )  # 修正

        # ========== 步骤 6: 构建代理损失 ==========
        target = (x0_vol + grad_corrected).detach()
        loss_sds = 0.5 * F.mse_loss(x0_vol, target, reduction="mean")

        # ========== 总损失 ==========
        loss_total = loss_sds

        return loss_total

    # def train_step_with_Fidelity(
    #     self,
    #     x0_vol,
    #     gt_projs,
    #     rays,
    #     bound_box,
    #     n_samples=512,
    #     rays_chunk=4096,
    #     fidelity_weight=0.05,
    #     step_ratio=None,
    #     slice_chunk=8,
    # ):
    #     """
    #     SDS + DPS 保真度项的训练步骤
        
    #     Args:
    #         x0_vol: [1, 1, D, H, W] 完整体积（from NAF）
    #         gt_projs: [N_rays] 或 [N_angles, H, W] GT 投影
    #         rays: [N_rays, 8] (rays_o, rays_d, near, far)
    #         bound_box: [3] 物理边界 (x,y,z)
    #         n_samples: int 每条光线的采样点数
    #         rays_chunk: int 每次投影处理的 rays 数量
    #         fidelity_weight: float λ 梯度修正权重
    #         step_ratio: float 用于 annealing 时间步调度
    #         slice_chunk: int 切片批大小，避免 OOM
            
    #     Returns:
    #         loss_total: scalar loss
    #     """
    #     if x0_vol.dim() != 5:
    #         raise ValueError(f"x0_vol must be [1,1,D,H,W], got {x0_vol.shape}")
    #     if rays.dim() != 2 or rays.shape[-1] != 8:
    #         raise ValueError(f"rays must be [N_rays,8], got {rays.shape}")
    #     if gt_projs.numel() != rays.shape[0]:
    #         raise ValueError(
    #             f"gt_projs size {gt_projs.numel()} must match N_rays {rays.shape[0]}"
    #         )
    #     if not torch.is_tensor(bound_box) or bound_box.numel() != 3:
    #         raise ValueError("bound_box must be a tensor with 3 elements [x,y,z]")
    #     if slice_chunk < 1:
    #         raise ValueError("slice_chunk must be >= 1")
    #     if n_samples < 2:
    #         raise ValueError("n_samples must be >= 2")
    #     if rays_chunk < 1:
    #         raise ValueError("rays_chunk must be >= 1")

    #     _, _, D, H, W = x0_vol.shape

    #     # ========== 步骤 1-4: 分块计算 score 与 x0_hat ==========
    #     eps = 1e-5
    #     t_full = (
    #         self.sample_t_annealing(D, step_ratio)
    #         if self.annealing
    #         else torch.rand(D, device=self.device) * (self.sde.T - eps) + eps
    #     )

    #     x0_slices = x0_vol.squeeze(0).squeeze(0)  # [D, H, W]
    #     grad_sds_list = []
    #     x0_hat_list = []
    #     sigma_t_list = []

    #     with torch.no_grad():
    #         score_fn = mutils.get_score_fn(
    #             self.sde, self.model, train=False, continuous=True
    #         )

    #         for i in range(0, D, slice_chunk):
    #             j = min(i + slice_chunk, D)
    #             x0_chunk = x0_slices[i:j].unsqueeze(1)  # [B, 1, H, W]
    #             t_chunk = t_full[i:j]

    #             z = torch.randn_like(x0_chunk)
    #             mean, std = self.sde.marginal_prob(x0_chunk, t_chunk)
    #             sigma_t = std[:, None, None, None]
    #             x_t = mean + sigma_t * z

    #             if self.use_paas and self.paas_k > 1:
    #                 score_pred = self._predict_score_paas(score_fn, x_t, t_chunk, sigma_t)
    #             else:
    #                 score_pred = score_fn(x_t, t_chunk)

    #             x0_hat = x_t + (sigma_t ** 2) * score_pred

    #             grad_sds = score_pred * sigma_t + z
    #             grad_sds_list.append(grad_sds)
    #             x0_hat_list.append(x0_hat)
    #             sigma_t_list.append(sigma_t)

    #     grad_sds = torch.cat(grad_sds_list, dim=0).unsqueeze(0)  # [1, D, 1, H, W] -> [1, D, 1, H, W]
    #     grad_sds = grad_sds.permute(0, 2, 1, 3, 4)  # [1, 1, D, H, W]

    #     x0_hat = torch.cat(x0_hat_list, dim=0).unsqueeze(0)
    #     x0_hat = x0_hat.permute(0, 2, 1, 3, 4)  # [1, 1, D, H, W]

    #     sigma_t_full = torch.cat(sigma_t_list, dim=0).unsqueeze(0)
    #     sigma_t_full = sigma_t_full.permute(0, 2, 1, 3, 4)  # [1, 1, D, H, W]
        
    #     # ========== 步骤 5a: Rays + grid_sample 正向投影 ==========
    #     x0_hat_proj = x0_hat.clone().detach().requires_grad_(True)
    #     proj_pred = self._project_volume_with_rays(
    #         x0_hat_proj, rays, bound_box, n_samples, rays_chunk
    #     )
        
    #     # ========== 步骤 5b: 计算投影残差梯度 ==========
    #     loss_fidelity_proj = 0.5 * F.mse_loss(proj_pred, gt_projs, reduction="sum")
        
    #     # 计算梯度 ∇_{x0_hat} ||y - A(x0_hat)||^2
    #     grad_fidelity = torch.autograd.grad(
    #         loss_fidelity_proj,
    #         x0_hat_proj,
    #         create_graph=False,
    #         retain_graph=False,
    #     )[0]  # [1, 1, D, H, W]
        
    #     # ========== 步骤 5c: 梯度修正 ==========
    #     grad_corrected = grad_sds - fidelity_weight * sigma_t_full * grad_fidelity  # 修正
        
    #     # ========== 步骤 6: 构建代理损失 ==========
    #     target = (x0_vol + grad_corrected).detach()
    #     loss_sds = 0.5 * F.mse_loss(x0_vol, target, reduction="mean")
        
    #     # ========== 总损失 ==========
    #     loss_total = loss_sds
        
    #     return loss_total
    
    def _project_volume_with_rays(self, volume, rays, bound_box, n_samples, rays_chunk):
        """
        使用 rays + grid_sample 对 volume 做可导投影。

        Args:
            volume: [1, 1, D, H, W]
            rays: [N_rays, 8] (o,d,near,far)
            bound_box: [3] (x,y,z)
            n_samples: int
        Returns:
            proj: [N_rays]
        """
        bound_tensor = bound_box if torch.is_tensor(bound_box) else torch.tensor(bound_box, device=rays.device)
        proj_chunks = []

        for i in range(0, rays.shape[0], rays_chunk):
            j = min(i + rays_chunk, rays.shape[0])
            rays_chunk_i = rays[i:j]

            rays_o = rays_chunk_i[..., :3]
            rays_d = rays_chunk_i[..., 3:6]
            near = rays_chunk_i[..., 6:7]
            far = rays_chunk_i[..., 7:8]

            t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=rays.device)
            z_vals = near * (1.0 - t_vals) + far * t_vals
            z_vals = z_vals.expand(rays_o.shape[0], n_samples)

            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts / bound_tensor
            pts = pts.clamp(-1.0, 1.0)

            grid = torch.stack([pts[..., 1], pts[..., 0], pts[..., 2]], dim=-1)
            grid = grid.unsqueeze(0).unsqueeze(3)

            samples = F.grid_sample(
                volume, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            samples = samples.squeeze(0).squeeze(0).squeeze(-1)

            dists = z_vals[..., 1:] - z_vals[..., :-1]
            last = torch.full_like(dists[..., :1], 1e-10)
            dists = torch.cat([dists, last], dim=-1)
            dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

            proj_chunks.append(torch.sum(samples * dists, dim=-1))

        return torch.cat(proj_chunks, dim=0)

    def _predict_score_paas(self, score_fn, x_t, t, sigma_t):
        """
        PAAS: 对 x_t 进行 K 次局部扰动后平均 score，降低方差。
        """
        k = self.paas_k
        perturb_scale = self.paas_rho * sigma_t
        score_sum = torch.zeros_like(x_t)

        # 逐次前向，避免一次性构造 [B*K, C, H, W] 造成显存峰值过高
        for _ in range(k):
            noise_p = torch.randn_like(x_t)
            x_t_pert = x_t + perturb_scale * noise_p
            score_sum = score_sum + score_fn(x_t_pert, t)

        return score_sum / float(k)

    def sample_t_annealing(self, batch_size, step_ratio):
        max_t_start = self.sde.T * 0.8
        max_t_end = self.sde.T * 0.2

        min_t_start = self.eps
        min_t_end = self.sde.T * 0.02

        current_max_t = max_t_start - step_ratio * (max_t_start - max_t_end)
        current_min_t = min_t_start + step_ratio * (min_t_end - min_t_start)

        return (
            torch.rand(batch_size, device=self.device) * (current_max_t - current_min_t)
            + current_min_t
        )
