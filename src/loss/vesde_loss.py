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
    def __init__(self, config_path, ckpt_path, device="cuda"):
        super().__init__()
        self.device = device

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

        target = (x0 - grad).detach()
        loss_sds = 0.5 * F.mse_loss(x0, target, reduction="mean")

        return loss_sds
