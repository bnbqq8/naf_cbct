import json
import os
import os.path as osp
from shutil import copyfile

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from .dataset import TIGREDataset as Dataset
from .encoder import get_encoder
from .loss.vesde_loss import VESDEGuidance
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
        # sds
        self.use_sds = cfg["train"].get("sds", False)
        if self.use_sds:
            self.sds_interval = cfg["train"]["sds"].get("interval", 1)
            self.sds_iterations = cfg["train"]["sds"].get("iterations", 1)
            self.sds_batch_size = cfg["train"]["sds"].get("batch_size", 8)
            self.sds_warmup_epochs = cfg["train"]["sds"].get("warmup_epochs", 0)
            self.lambda_sds = cfg["train"]["sds"].get("sds_weight", 1.0)
            self.img_dims = cfg["train"]["sds"].get("img_dims", [256, 256, 173])
            spacing = cfg["train"]["sds"].get("spacing", 1.0)
            self.bound_z = self.img_dims[2] * spacing / 2.0 /1000.0 # mm to m
            self.bound_xy = self.img_dims[0] * spacing / 2.0 /1000.0 # mm to m
            self.device = device
            # prepare vesde
            ckpt_path = cfg["train"]["sds"].get("ckpt_path", None)
            config_path = cfg["train"]["sds"].get("config_path", None)
            if ckpt_path is None or config_path is None:
                raise ValueError("SDS enabled but 'ckpt_path' or 'config_path' is missing in config.")
            self.vesde_guidance = VESDEGuidance(config_path, ckpt_path, device=device)
        # Log direcotry
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)

        # Dataset
        train_dset = Dataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "train", device)
        self.eval_dset = Dataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "val", device) if self.i_eval > 0 else None
        self.train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=cfg["train"]["n_batch"])
        self.voxels = self.eval_dset.voxels if self.i_eval > 0 else None
    
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
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999))
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer, gamma=cfg["train"]["lrate_gamma"])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])

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
        pbar = tqdm(total= iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start*iter_per_epoch)

        for idx_epoch in range(self.epoch_start, self.epochs+1):

            # Evaluate
            if (idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs) and self.i_eval > 0:
                self.net.eval()
                with torch.no_grad():
                    loss_test = self.eval_step(global_step=self.global_step, idx_epoch=idx_epoch)
                self.net.train()
                tqdm.write(f"[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}")
            
            # Train
            for data in self.train_dloader:
                self.global_step += 1
                # Train
                self.net.train()
                loss_train = self.train_step(data, global_step=self.global_step, idx_epoch=idx_epoch)
                #-----------------sds-----------------
                loss_sds = float("nan")
                if self.use_sds and idx_epoch > self.sds_warmup_epochs:
                    if idx_epoch % self.sds_interval == 0:
                        loss_sds = self.train_step_sds(global_step=self.global_step, idx_epoch=idx_epoch)
                    # loss_train_sds_sum += loss_sds
                #-----------------sds-----------------
                # pbar.set_description(f"epoch={idx_epoch}/{self.epochs}, loss={loss_train:.3g}, lr={self.optimizer.param_groups[0]['lr']:.3g}")
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_description(
                    f"Ep:{idx_epoch}/{self.epochs} | "
                    f"L_data:{loss_train:.4f} | "
                    f"L_sds:{loss_sds:.4f} | "
                    f"LR:{current_lr:.2e}"
                )
                pbar.update(1)
            
            # Save
            if (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs) and self.i_save > 0 and idx_epoch > 0:
                if osp.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                )

            # Update lrate
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
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
        修正后的 SDS 训练步：全切片覆盖 + 多轮迭代
        """
        total_sds_loss = 0.0
        updates_count = 0
        
        # 获取体积的总层数，例如 256
        D = self.img_dims[2] 
        
        # 设置 SDS 的 Batch Size (显存允许的情况下尽量大，比如 4, 8, 16)
        # 建议在 config 里配置 self.sds_batch_size
        batch_size = getattr(self, 'sds_batch_size', 8)
        
        # 设置 SDS 内部迭代次数
        iterations = getattr(self, 'sds_iterations', 1)

        # ===========================
        # 需求 2: sds_iteration 循环
        # ===========================
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
                step_ratio = global_step / self.max_steps if hasattr(self, 'max_steps') else 0.5
                
                # 注意：vesde_guidance 需要能处理 batch 输入 [B, 1, H, W]
                # 大多数实现都支持自动 broadcasting
                loss_sds_val = self.vesde_guidance.train_step(pred_slices, step_ratio=step_ratio)
                
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
    
    def sample_slice_batch(self, z_indices, res=256):
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
        D = self.img_dims[2] 
        z_vals = -self.bound_z + (z_indices / (D - 1.0)) * 2.0 * self.bound_z
        z_vals = z_vals.to(self.device).view(-1, 1, 1) # [B, 1, 1]

        # 2. 构建 XY 网格 (所有切片共用)
        coords = torch.linspace(-self.bound_xy, self.bound_xy, res, device=self.device)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij') # [H, W]
        
        # 3. 扩展为 Batch 坐标
        # grid_x, grid_y: [H, W] -> [B, H, W]
        B = z_indices.shape[0]
        batch_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        batch_y = grid_y.unsqueeze(0).expand(B, -1, -1)
        batch_z = z_vals.expand(B, res, res) # [B, H, W]
        
        # 堆叠坐标: [B, H, W, 3]
        coords_3d = torch.stack([batch_x, batch_y, batch_z], dim=-1)
        
        # 4. 展平并查询网络
        coords_flat = coords_3d.reshape(-1, 3) # [B*H*W, 3]
        
        # 使用你原有的 run_network
        from src.render.render import run_network 
        density_flat = run_network(coords_flat, self.net, self.netchunk)
        
        # 5. 重塑与归一化
        pred_slices = density_flat.reshape(B, 1, res, res)
        
        # 归一化 (根据你的 mu_water 调整)
        scale_factor = 1.0 / (self.mu_water * 1.2) if hasattr(self, 'mu_water') else 10.0
        pred_slices_norm = pred_slices * scale_factor
        
        return pred_slices_norm
    
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
        