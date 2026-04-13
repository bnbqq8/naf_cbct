import json
import os
import os.path as osp
from calendar import c
from shutil import copyfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from .dataset import TIGREDataset as Dataset
from .encoder import get_encoder
from .loss.vesde_loss import VESDEGuidance
from .models import utils as mutils
from .network import get_network


class Trainer:
    def __init__(self, cfg, device="cuda"):

        # Args
        self.global_step = 0
        self.conf = cfg
        self.n_fine = cfg["render"]["n_fine"]
        self.epochs = cfg["train"]["epoch"]
        self.i_eval = cfg["log"]["i_eval"]
        self.i_save = cfg["log"]["i_save"]
        self.netchunk = cfg["render"]["netchunk"]
        self.n_rays = cfg["train"]["n_rays"]
        self.device = device
        # sds
        self.use_sds = cfg["train"].get("sds", False)
        if self.use_sds:
            self.sds_interval = cfg["train"]["sds"].get("interval", 1)
            self.sds_iterations = cfg["train"]["sds"].get("iterations", 1)
            self.sds_batch_size = cfg["train"]["sds"].get("batch_size", 8)
            self.sds_warmup_epochs = cfg["train"]["sds"].get("warmup_epochs", 0)
            self.lambda_sds = cfg["train"]["sds"].get("sds_weight", 1.0)
            self.use_fidelity = cfg["train"]["sds"].get("use_fidelity", False)
            self.fidelity_weight = cfg["train"]["sds"].get("fidelity_weight", 0.05)
            self.fidelity_start_epoch = cfg["train"]["sds"].get("fidelity_start_epoch", 0)
            self.fidelity_slice_chunk = cfg["train"]["sds"].get("fidelity_slice_chunk", 8)
            # prepare vesde
            ckpt_path = cfg["train"]["sds"].get("ckpt_path", None)
            config_path = cfg["train"]["sds"].get("config_path", None)
            if ckpt_path is None or config_path is None:
                raise ValueError(
                    "SDS enabled but 'ckpt_path' or 'config_path' is missing in config."
                )
            self.vesde_guidance = VESDEGuidance(
                config_path,
                ckpt_path,
                annealing=cfg["train"]["sds"].get("annealing", False),
                device=device,
                use_paas=cfg["train"]["sds"].get("use_paas", False),
                paas_k=cfg["train"]["sds"].get("paas_k", 1),
                paas_rho=cfg["train"]["sds"].get("paas_rho", 0.1),
            )
            self.annealing = self.vesde_guidance.annealing
            self.sde = self.vesde_guidance.sde
            self.model = self.vesde_guidance.model
            self.use_paas = self.vesde_guidance.use_paas
            self.paas_k = self.vesde_guidance.paas_k
            self._predict_score_paas = self.vesde_guidance._predict_score_paas
            self.sample_t_annealing = self.vesde_guidance.sample_t_annealing
            self.sag = cfg["train"]["sds"].get("sag", None)
            if self.sag is not None:
                config_path = self.sag.get("config_path", None)
                ckpt_path = self.sag.get("ckpt_path", None)
                if ckpt_path is None or config_path is None:
                    raise ValueError(
                        "SAG enabled but 'ckpt_path' or 'config_path' is missing in config."
                    )
                self.vesde_guidance_sag = VESDEGuidance(
                    config_path,
                    ckpt_path,
                    annealing=cfg["train"]["sds"].get("annealing", False),
                    device=device,
                    use_paas=self.sag.get(
                        "use_paas", cfg["train"]["sds"].get("use_paas", False)
                    ),
                    paas_k=self.sag.get(
                        "paas_k", cfg["train"]["sds"].get("paas_k", 1)
                    ),
                    paas_rho=self.sag.get(
                        "paas_rho", cfg["train"]["sds"].get("paas_rho", 0.1)
                    ),
                )
        # Log direcotry
        self.expdir = osp.join(
            cfg["exp"]["expdir"], cfg["exp"]["expname"], cfg["exp"]["patient_id"]
        )
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)

        # Dataset
        train_dset = Dataset(
            cfg["exp"]["datadir"], cfg["train"]["n_rays"], "train", device
        )
        self.eval_dset = (
            Dataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "val", device)
            if self.i_eval > 0
            else None
        )
        self.train_dloader = torch.utils.data.DataLoader(
            train_dset, batch_size=cfg["train"]["n_batch"]
        )
        self.voxels = self.eval_dset.voxels if self.i_eval > 0 else None
        # Boundary
        self.img_dims = self.eval_dset.image.shape  # [256, 256, 173]
        spacing = self.eval_dset.dVoxel[0]  # cfg["train"]["sds"].get("spacing", 1.0)
        self.bound_z = self.img_dims[2] * spacing / 2.0 / 1000.0  # mm to m
        self.bound_xy = self.img_dims[0] * spacing / 2.0 / 1000.0  # mm to m
        self.bound_box = torch.tensor(
            [self.bound_xy, self.bound_xy, self.bound_z], device=device
        )

        # Network
        network = get_network(cfg["network"]["net_type"])
        cfg["network"].pop("net_type", None)
        encoder = get_encoder(**cfg["encoder"])
        self.net = network(encoder, **cfg["network"]).to(device)
        grad_vars = list(self.net.parameters())
        self.net_fine = None
        if self.n_fine > 0:
            self.net_fine = network(encoder, **cfg["network"]).to(device)
            grad_vars += list(self.net_fine.parameters())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999)
        )
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer, gamma=cfg["train"]["lrate_gamma"])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=cfg["train"]["lrate_step"],
            gamma=cfg["train"]["lrate_gamma"],
        )

        # Load checkpoints
        self.epoch_start = 0
        if cfg["train"]["resume"] and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start * len(self.train_dloader)
            self.net.load_state_dict(ckpt["network"])
            if self.n_fine > 0:
                self.net_fine.load_state_dict(ckpt["network_fine"])

        # Summary writer
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)

    def args2string(self, hp):
        """
        Transfer args to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def start(self):
        """
        Main loop.
        """

        def fmt_loss_str(losses):
            return "".join(", " + k + ": " + f"{losses[k].item():.3g}" for k in losses)

        iter_per_epoch = len(self.train_dloader)
        pbar = tqdm(total=iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start * iter_per_epoch)

        for idx_epoch in range(self.epoch_start, self.epochs + 1):

            # Evaluate
            if (
                idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs
            ) and self.i_eval > 0:
                self.net.eval()
                with torch.no_grad():
                    loss_test = self.eval_step(
                        global_step=self.global_step, idx_epoch=idx_epoch
                    )
                self.net.train()
                tqdm.write(
                    f"[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}"
                )

            # Train
            for data in self.train_dloader:
                self.global_step += 1
                # Train
                self.net.train()
                loss_train = self.train_step(
                    data, global_step=self.global_step, idx_epoch=idx_epoch
                )
                # -----------------sds-----------------
                loss_sds = float("nan")
                if self.use_sds and idx_epoch > self.sds_warmup_epochs:
                    if idx_epoch % self.sds_interval == 0:
                        if self.use_fidelity and idx_epoch >= self.fidelity_start_epoch:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            step_ratio = (idx_epoch - self.sds_warmup_epochs) / (
                                self.epochs - self.sds_warmup_epochs
                            )
                            step_ratio = max(0.0, min(1.0, step_ratio))

                            fidelity_res = (
                                self.conf.get("train", {})
                                .get("sds", {})
                                .get("fidelity_res", 256)
                            )
                            offload_diffusion = (
                                self.conf.get("train", {})
                                .get("sds", {})
                                .get("fidelity_offload_diffusion", True)
                            )
                            offload_naf = (
                                self.conf.get("train", {})
                                .get("sds", {})
                                .get("fidelity_offload_naf", False)
                            )
                            if self.conf.get("encoder", {}).get("encoding") == "hashgrid":
                                offload_naf = False
                            if offload_diffusion:
                                self.vesde_guidance.model.to("cpu")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            if offload_naf:
                                self.net.to("cpu")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            x0_vol = self.sample_volume(res=fidelity_res, direction="ax")
                            if offload_naf:
                                self.net.to(self.device)
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            if offload_diffusion:
                                self.vesde_guidance.model.to(self.device)
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            x0_vol = x0_vol.float()
                            eval_index = self.conf["train"]["sds"].get(
                                "fidelity_eval_index", 0
                            )
                            rays = (
                                self.eval_dset.rays[eval_index]
                                .reshape(-1, 8)
                                .to(self.device)
                            )
                            gt_projs = (
                                self.eval_dset.projs[eval_index]
                                .reshape(-1)
                                .to(self.device)
                            )

                            n_samples = self.conf["render"]["n_samples"]
                            rays_chunk = self.conf["train"]["sds"].get(
                                "fidelity_rays_chunk", 4096
                            )

                            loss_sds = self.train_step_sds_with_fidelity(
                                x0_vol,
                                gt_projs,
                                rays,
                                self.bound_box,
                                n_samples=n_samples,
                                rays_chunk=rays_chunk,
                                fidelity_weight=self.fidelity_weight,
                                step_ratio=step_ratio,
                                slice_chunk=self.fidelity_slice_chunk,
                            )
                        else:
                            loss_sds = self.train_step_sds(
                                global_step=self.global_step, idx_epoch=idx_epoch
                            )
                    # loss_train_sds_sum += loss_sds
                # -----------------sds-----------------
                # pbar.set_description(f"epoch={idx_epoch}/{self.epochs}, loss={loss_train:.3g}, lr={self.optimizer.param_groups[0]['lr']:.3g}")
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_description(
                    f"Ep:{idx_epoch}/{self.epochs} | "
                    f"L_data:{loss_train:.4f} | "
                    f"L_sds:{loss_sds:.4f} | "
                    f"LR:{current_lr:.2e}"
                )
                pbar.update(1)

            # Save
            if (
                (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs)
                and self.i_save > 0
                and idx_epoch > 0
            ):
                if osp.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(
                    f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}"
                )
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "network_fine": (
                            self.net_fine.state_dict() if self.n_fine > 0 else None
                        ),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                )

            # Update lrate
            self.writer.add_scalar(
                "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
            )
            self.lr_scheduler.step()

        tqdm.write(f"Training complete! See logs in {self.expdir}")

    def train_step(self, data, global_step, idx_epoch):
        """
        Training step
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, global_step, idx_epoch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step_sds(self, global_step, idx_epoch):
        """
        修正后的 SDS 训练步：全切片覆盖 + 多轮迭代 + 可选 Fidelity 项
        """
        total_sds_loss = 0.0
        updates_count = 0

        # 获取体积的总层数，例如 256
        W, _, D = self.img_dims

        # 设置 SDS 的 Batch Size (显存允许的情况下尽量大，比如 4, 8, 16)
        # 建议在 config 里配置 self.sds_batch_size
        batch_size = getattr(self, "sds_batch_size", 8)

        # 设置 SDS 内部迭代次数
        iterations = getattr(self, "sds_iterations", 1)
        
        # 计算当前训练进度的比例，用于时间步采样的 annealing
        step_ratio = (idx_epoch - self.sds_warmup_epochs) / (
            self.epochs - self.sds_warmup_epochs
        )
        step_ratio = max(0.0, min(1.0, step_ratio))

        # ========== 传统切片 SDS 方法 ==========
        # ===========================
        # 需求 2: sds_iteration 循环
        # ===========================
        # ax
        for it in range(iterations):

            # 生成所有切片的索引列表 [0, 1, ..., D-1]
            all_indices = torch.arange(D, device=self.device)

            # 打乱顺序 (Shuffle)，这对 SGD 很重要，避免网络记住了切片顺序
            perm = torch.randperm(D, device=self.device)
            all_indices = all_indices[perm]

            # ===========================
            # 需求 1: 遍历所有 Slices (Mini-batch Loop)
            # ===========================
            # 将索引按 batch_size 切分
            batches = torch.split(all_indices, batch_size)

            for batch_indices in batches:
                # 1. 清空梯度 (每个 mini-batch 都要更新一次参数)
                self.optimizer.zero_grad()

                # 2. 采样指定的切片 Batch
                # batch_indices 是一个 tensor, 如 [0, 15, 32, ...]
                pred_slices = self.sample_slice_batch(batch_indices, res=256)

                # 3. 计算 SDS Loss
                # 注意：vesde_guidance 需要能处理 batch 输入 [B, 1, H, W]
                # 大多数实现都支持自动 broadcasting
                loss_sds_val = self.vesde_guidance.train_step(
                    pred_slices, step_ratio=step_ratio
                )

                # 4. 加权
                # loss_sds_val 通常已经是 batch 的平均值
                loss_final = self.lambda_sds * loss_sds_val

                # 5. 反向传播与更新
                loss_final.backward()
                self.optimizer.step()

                # 统计 Loss
                total_sds_loss += loss_sds_val.item()
                updates_count += 1

        # sag
        if self.sag is not None:
            all_indices = torch.arange(W, device=self.device)
            perm = torch.randperm(W, device=self.device)
            all_indices = all_indices[perm]
            batches = torch.split(all_indices, batch_size)
            for it in range(iterations):
                # 生成所有切片的索引列表 [0, 1, ..., D-1]
                all_indices = torch.arange(W, device=self.device)

                # 打乱顺序 (Shuffle)，这对 SGD 很重要，避免网络记住了切片顺序
                perm = torch.randperm(W, device=self.device)
                all_indices = all_indices[perm]

                # ===========================
                # 需求 1: 遍历所有 Slices (Mini-batch Loop)
                # ===========================
                # 将索引按 batch_size 切分
                batches = torch.split(all_indices, batch_size)

                for batch_indices in batches:
                    # 1. 清空梯度 (每个 mini-batch 都要更新一次参数)
                    self.optimizer.zero_grad()

                    # 2. 采样指定的切片 Batch
                    # batch_indices 是一个 tensor, 如 [0, 15, 32, ...]
                    pred_slices = self.sample_slice_batch(
                        batch_indices, res=256, direction="sag"
                    )
                    # 2.1. pad to (256, 256)
                    pred_slices = self.pad_to_size(pred_slices, target_size=(256, 256))

                    # 3. 计算 SDS Loss
                    step_ratio = (
                        global_step / self.max_steps
                        if hasattr(self, "max_steps")
                        else 0.5
                    )

                    # 注意：vesde_guidance 需要能处理 batch 输入 [B, 1, H, W]
                    # 大多数实现都支持自动 broadcasting
                    loss_sds_val = self.vesde_guidance_sag.train_step(
                        pred_slices, step_ratio=step_ratio
                    )

                    # 4. 加权
                    # loss_sds_val 通常已经是 batch 的平均值
                    loss_final = self.lambda_sds * loss_sds_val

                    # 5. 反向传播与更新
                    loss_final.backward()
                    self.optimizer.step()

                    # 统计 Loss
                    total_sds_loss += loss_sds_val.item()
                    updates_count += 1

        # 返回平均 Loss
        return total_sds_loss / max(updates_count, 1)

    # def train_step_sds_with_fidelity_old(self, data, global_step, idx_epoch):
    #     """
    #     使用 rays + grid_sample 的 Fidelity SDS 训练步。
    #     """
    #     step_ratio = (idx_epoch - self.sds_warmup_epochs) / (
    #         self.epochs - self.sds_warmup_epochs
    #     )
    #     step_ratio = max(0.0, min(1.0, step_ratio))

    #     x0_vol = self.sample_volume(res=256, direction="ax")
    #     eval_index = self.conf["train"]["sds"].get("fidelity_eval_index", 0)
    #     rays = self.eval_dset.rays[eval_index].reshape(-1, 8).to(self.device)
    #     gt_projs = self.eval_dset.projs[eval_index].reshape(-1).to(self.device)

    #     n_samples = self.conf["render"]["n_samples"]
    #     rays_chunk = self.conf["train"]["sds"].get("fidelity_rays_chunk", 4096)

    #     self.optimizer.zero_grad()
    #     loss_sds_val = self.vesde_guidance.train_step_with_Fidelity(
    #         x0_vol,
    #         gt_projs,
    #         rays,
    #         self.bound_box,
    #         n_samples=n_samples,
    #         rays_chunk=rays_chunk,
    #         fidelity_weight=self.fidelity_weight,
    #         step_ratio=step_ratio,
    #         slice_chunk=self.fidelity_slice_chunk,
    #     )

    #     loss_fidelity = self.lambda_sds * loss_sds_val
    #     loss_fidelity.backward()
    #     self.optimizer.step()

    #     return loss_sds_val.item()
    #     # sag
    #     if self.sag is not None:
    #         all_indices = torch.arange(W, device=self.device)
    #         perm = torch.randperm(W, device=self.device)
    #         all_indices = all_indices[perm]
    #         batches = torch.split(all_indices, batch_size)
    #         for it in range(iterations):
    #             # 生成所有切片的索引列表 [0, 1, ..., D-1]
    #             all_indices = torch.arange(W, device=self.device)

    #             # 打乱顺序 (Shuffle)，这对 SGD 很重要，避免网络记住了切片顺序
    #             perm = torch.randperm(W, device=self.device)
    #             all_indices = all_indices[perm]

    #             # ===========================
    #             # 需求 1: 遍历所有 Slices (Mini-batch Loop)
    #             # ===========================
    #             # 将索引按 batch_size 切分
    #             batches = torch.split(all_indices, batch_size)

    #             for batch_indices in batches:
    #                 # 1. 清空梯度 (每个 mini-batch 都要更新一次参数)
    #                 self.optimizer.zero_grad()

    #                 # 2. 采样指定的切片 Batch
    #                 # batch_indices 是一个 tensor, 如 [0, 15, 32, ...]
    #                 pred_slices = self.sample_slice_batch(
    #                     batch_indices, res=256, direction="sag"
    #                 )
    #                 # 2.1. pad to (256, 256)
    #                 pred_slices = self.pad_to_size(pred_slices, target_size=(256, 256))

    #                 # 3. 计算 SDS Loss
    #                 step_ratio = (
    #                     global_step / self.max_steps
    #                     if hasattr(self, "max_steps")
    #                     else 0.5
    #                 )

    #                 # 注意：vesde_guidance 需要能处理 batch 输入 [B, 1, H, W]
    #                 # 大多数实现都支持自动 broadcasting
    #                 loss_sds_val = self.vesde_guidance_sag.train_step(
    #                     pred_slices, step_ratio=step_ratio
    #                 )

    #                 # 4. 加权
    #                 # loss_sds_val 通常已经是 batch 的平均值
    #                 loss_final = self.lambda_sds * loss_sds_val

    #                 # 5. 反向传播与更新
    #                 loss_final.backward()
    #                 self.optimizer.step()

    #                 # 统计 Loss
    #                 total_sds_loss += loss_sds_val.item()
    #                 updates_count += 1
    #     # 返回平均 Loss
    #     return total_sds_loss / max(updates_count, 1)
    
    def train_step_sds_with_fidelity(
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
            self.vesde_guidance.sample_t_annealing(D, step_ratio)
            if self.vesde_guidance.annealing
            else torch.rand(D, device=self.device)
            * (self.vesde_guidance.sde.T - eps)
            + eps
        )

        x0_slices = x0_vol.squeeze(0).squeeze(0).unsqueeze(1)  # [D, 1, H, W]
        z = torch.randn_like(x0_slices)
        mean, std = self.vesde_guidance.sde.marginal_prob(x0_slices, t_full)
        sigma_t = std[:, None, None, None]
        x_t = mean + sigma_t * z

        x_t.requires_grad_(True)

        grad_sds_list = []
        x0_hat_list = []
        sigma_t_list = []

        # with torch.no_grad():
        score_fn = mutils.get_score_fn(
            self.vesde_guidance.sde,
            self.vesde_guidance.model,
            train=False,
            continuous=True,
        )

        for i in range(0, D, slice_chunk):
            j = min(i + slice_chunk, D)
            # x0_chunk = x0_slices[i:j].unsqueeze(1)  # [B, 1, H, W]
            x_t_chunk = x_t[i:j]
            t_chunk = t_full[i:j]
            sigma_t_chunk = sigma_t[i:j]
            z_chunk = z[i:j]

            with torch.no_grad():
                if (
                    self.vesde_guidance.use_paas
                    and self.vesde_guidance.paas_k > 1
                ):
                    score_pred = self.vesde_guidance._predict_score_paas(
                        score_fn, x_t_chunk, t_chunk, sigma_t_chunk
                    )
                else:
                    score_pred = score_fn(x_t_chunk, t_chunk)

            x0_hat = x_t_chunk + (sigma_t_chunk**2) * score_pred

            grad_sds = score_pred * sigma_t_chunk + z_chunk
            grad_sds_list.append(grad_sds)
            x0_hat_list.append(x0_hat)
            sigma_t_list.append(sigma_t_chunk)

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

        proj_pred = self.vesde_guidance._project_volume_with_rays(
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
        )[0]
        if grad_fidelity.dim() == 4:
            grad_fidelity = grad_fidelity.unsqueeze(0).permute(0, 2, 1, 3, 4)

        # ========== 步骤 5c: 梯度修正 ==========
        grad_sds.addcmul_(sigma_t_full, grad_fidelity, value=-fidelity_weight)
        grad_corrected = grad_sds  # 修正

        # ========== 步骤 6: 构建代理损失 ==========
        target = (x0_vol + grad_corrected).detach()
        loss_sds = 0.5 * F.mse_loss(x0_vol, target, reduction="mean")

        # ========== 总损失 ==========
        loss_total = loss_sds

        return loss_total

    def sample_slice_batch(self, z_indices, res=256, direction="ax"):
        """
        根据指定的 z 轴索引生成切片 Batch
        Args:
            z_indices (Tensor): [B], 包含要采样的层索引 (0 ~ D-1)
            res (int): 切片分辨率
        Returns:
            pred_slice (Tensor): [B, 1, H, W]
        """
        # 1. 将整数索引映射到物理坐标 z (假设 z 范围是 [-bound, bound])
        # NAF 通常定义在 [-self.bound, self.bound]
        # 公式: -bound + (idx / (D-1)) * 2 * bound
        # 假设 self.img_dims[2] 是 Z 轴的总层数 (e.g., 256)
        D = self.img_dims[2] if direction == "ax" else self.img_dims[0]
        bound_slice = self.bound_z if direction == "ax" else self.bound_xy
        z_vals = -bound_slice + (z_indices / (D - 1.0)) * 2.0 * bound_slice
        z_vals = z_vals.to(self.device).view(-1, 1, 1)  # [B, 1, 1]
        B = z_indices.shape[0]
        # 2. 构建 XY 网格 (所有切片共用)
        if direction == "ax":
            coords = torch.linspace(
                -self.bound_xy, self.bound_xy, res, device=self.device
            )
            grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")  # [H, W]
            # 3. 扩展为 Batch 坐标
            # grid_x, grid_y: [H, W] -> [B, H, W]
            batch_x = grid_x.unsqueeze(0).expand(B, -1, -1)
            batch_y = grid_y.unsqueeze(0).expand(B, -1, -1)
            batch_z = z_vals.expand(B, res, res)  # [B, H, W]

            # 堆叠坐标: [B, H, W, 3]
            coords_3d = torch.stack([batch_x, batch_y, batch_z], dim=-1)
        elif direction == "sag":
            coords_y = torch.linspace(
                -self.bound_xy, self.bound_xy, res, device=self.device
            )
            coords_z = torch.linspace(
                -self.bound_z, self.bound_z, self.img_dims[2], device=self.device
            )
            grid_y, grid_z = torch.meshgrid(coords_y, coords_z, indexing="ij")  # [H, W]
            batch_y = grid_y.unsqueeze(0).expand(B, -1, -1)
            batch_z = grid_z.unsqueeze(0).expand(B, -1, -1)
            batch_x = z_vals.expand(B, res, self.img_dims[2])  # [B, H, W]

            coords_3d = torch.stack([batch_x, batch_y, batch_z], dim=-1)
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        # 4. 展平并查询网络
        coords_flat = coords_3d.reshape(-1, 3)  # [B*H*W, 3]

        # normalizing points coords
        coords_flat = coords_flat / self.bound_box  # normalize to [-1, 1]

        # 使用你原有的 run_network
        from src.render.render import run_network
        netchunk = min(self.netchunk, 128)
        amp_enabled = (
            self.conf.get("train", {})
            .get("sds", {})
            .get("fidelity_amp", True)
        )

        density_flat = run_network(coords_flat, self.net, self.netchunk)

        # 5. 重塑与归一化
        pred_slices = (
            density_flat.reshape(B, 1, res, res)
            if direction == "ax"
            else density_flat.reshape(B, 1, res, self.img_dims[2])
        )

        # # 归一化 (根据你的 mu_water 调整)
        # scale_factor = 1.0 / (self.mu_water * 1.2) if hasattr(self, 'mu_water') else 10.0
        # pred_slices_norm = pred_slices * scale_factor

        return pred_slices

    def compute_loss(self, data, global_step, idx_epoch):
        """
        Training step
        """
        raise NotImplementedError()

    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        raise NotImplementedError()

    def sample_volume(self, res=256, direction="ax"):
        """
        从 NAF MLP 查询完整 3D 体积
        
        Args:
            res: 横向分辨率 (H, W)
            direction: "ax" (axial) 或 "sag" (sagittal)
        
        Returns:
            volume: [1, 1, D, H, W] 完整体积
        """
        D = self.img_dims[2] if direction == "ax" else self.img_dims[0]
        H = W = res
        
        # 1. 按切片生成坐标并逐片查询，降低显存峰值
        bound_z = self.bound_z if direction == "ax" else self.bound_xy
        volume_slices = []

        from src.render.render import run_network
        netchunk = min(self.netchunk, 512)
        amp_enabled = (
            self.conf.get("train", {})
            .get("sds", {})
            .get("fidelity_amp", True)
        )

        for z_idx in range(D):
            z_val = -bound_z + (z_idx / (D - 1.0)) * 2.0 * bound_z

            if direction == "ax":
                coords = torch.linspace(
                    -self.bound_xy, self.bound_xy, res, device=self.device
                )
                grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
                grid_z = torch.full_like(grid_x, z_val)
                coords_3d = torch.stack([grid_x, grid_y, grid_z], dim=-1)
            else:  # sag
                coords_y = torch.linspace(
                    -self.bound_xy, self.bound_xy, res, device=self.device
                )
                coords_z = torch.linspace(
                    -self.bound_z, self.bound_z, D, device=self.device
                )
                grid_y, grid_z = torch.meshgrid(coords_y, coords_z, indexing="ij")
                grid_x = torch.full_like(grid_y, z_val)
                coords_3d = torch.stack([grid_x, grid_y, grid_z], dim=-1)

            coords_flat = coords_3d.reshape(-1, 3)
            coords_flat = coords_flat / self.bound_box
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    density_flat = run_network(coords_flat, self.net, netchunk)
            volume_slices.append(density_flat.reshape(H, W))

        volume = torch.stack(volume_slices, dim=0)
        volume = volume.unsqueeze(0).unsqueeze(0)

        return volume

    def pad_to_size(self, x, target_size=(256, 256)):
        """
        将输入张量 x 填充到目标大小，支持梯度回传
        """
        *batch_dims, h, w = x.shape
        th, tw = target_size

        # 创建画布
        canvas = torch.zeros((*batch_dims, th, tw), device=x.device, dtype=x.dtype)
        start_h = (th - h) // 2
        start_w = (tw - w) // 2
        # 如果想靠左上角对齐：
        canvas[..., start_h : start_h + h, start_w : start_w + w] = x
        return canvas
