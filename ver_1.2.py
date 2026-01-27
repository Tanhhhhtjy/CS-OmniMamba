# 这是优化的1.2版本
'''
该版本只有关于损失函数的修改
尝试引入"频域增强的连续物理加权损失"
'''
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
# ==========================================
# 0. 全局配置与路径
# ==========================================
SAVE_FIGURES_DIR = "./results_ver1.2"
os.makedirs(SAVE_FIGURES_DIR, exist_ok=True)

# 核心参数配置
CONFIG = {
    'img_size': 66,
    'img_size_w': 70,
    'patch_size': 2,    # 优化点
    'stride': 2,        # 优化点
    'd_state': 32,      # 减少过拟合
    'dim': 128,
    'depth': 4,
    'batch_size': 8,
    'epochs': 1000,      # 训练轮数
    'lr': 1e-4
}

# ==========================================
# 1. 模型架构 
# ==========================================

class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
      
        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate_fc[2].bias, -2.0)
        self.last_gate_mean = None

    def forward(self, x_q, x_kv):
        context, _ = self.attn(self.norm_q(x_q), self.norm_kv(x_kv), self.norm_kv(x_kv))
        concat = torch.cat([x_q, context], dim=-1)
        gate = self.gate_fc(concat)
      
        # 记录 Gate 值用于绘图
        if self.training:
             try:
                 self.last_gate_mean = gate.mean().detach().cpu().item()
             except:
                 self.last_gate_mean = None
      
        output = x_q + gate * (context - x_q)
        return output

class OmniBiMambaBlock_Pseudo(nn.Module):
    """
    全向扫描伪 Mamba 模块：同时进行水平和垂直方向的序列建模
    """
    def __init__(self, dim, d_state=64, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        self.norm = nn.LayerNorm(dim)
        
        # 输入投影，生成 x 和 gate (z)
        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        
        # 1D 卷积，用于提取局部特征
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, 
                                bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.activation = nn.SiLU()

        # 需要两个 SSM 分支
        
        # 分支 1: 水平扫描 (Horizontal)
        self.ssm_h = nn.GRU(
            input_size=self.d_inner,
            hidden_size=d_state, 
            num_layers=1,
            batch_first=True,
            bidirectional=True 
        )
        self.ssm_proj_h = nn.Linear(d_state * 2, self.d_inner) # GRU双向输出映射回 d_inner

        # 分支 2: 垂直扫描 (Vertical)
        self.ssm_v = nn.GRU(
            input_size=self.d_inner,
            hidden_size=d_state, 
            num_layers=1,
            batch_first=True,
            bidirectional=True 
        )
        self.ssm_proj_v = nn.Linear(d_state * 2, self.d_inner)

        # 融合层:将水平与垂直特征拼接后进行融合
        self.fusion_linear = nn.Linear(self.d_inner * 2, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, H, W):
        """
        x: [Batch, Length, Dim] 其中 Length = H * W
        H, W: 当前 Patch 特征图的高和宽
        """
        residual = x
        x = self.norm(x)
        
        # 1. 投影与门控切分
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)
        
        # 2. 局部卷积处理
        x_branch = x_branch.permute(0, 2, 1) # [B, D, L]
        x_branch = self.conv1d(x_branch)[:, :, :x.shape[1]]
        x_branch = self.activation(x_branch)
        x_branch = x_branch.permute(0, 2, 1) # [B, L, D]
        
        # 3. 全向扫描 (核心逻辑)
        
        # 3.1 水平流 (保持原样)
        # x_branch: [B, H*W, D]
        ssm_out_h, _ = self.ssm_h(x_branch)
        x_feat_h = self.ssm_proj_h(ssm_out_h)
        
        # 3.2 垂直流 (转置 -> 扫描 -> 还原)
        B, L, C = x_branch.shape
        # View成2D -> 转置 -> 展平
        x_v_in = x_branch.view(B, H, W, C).permute(0, 2, 1, 3).contiguous().view(B, L, C)
        
        ssm_out_v, _ = self.ssm_v(x_v_in)
        x_feat_v = self.ssm_proj_v(ssm_out_v)
        
        # 还原回原来的顺序: View成2D(此时是W,H) -> 转置回(H,W) -> 展平
        x_feat_v = x_feat_v.view(B, W, H, C).permute(0, 2, 1, 3).contiguous().view(B, L, C)
        
        # 3.3 融合水平和垂直特征 (自适应融合拼接)
        x_combined = torch.cat([x_feat_h, x_feat_v], dim=-1)  # [B, L, 2*C]
        x_ssm_total = self.fusion_linear(x_combined)  # [B, L, C]

        # 4. 门控相乘
        z_branch = self.activation(z_branch)
        x_out = x_ssm_total * z_branch
        
        # 5. 输出投影
        out = self.out_proj(x_out)
        return residual + self.dropout(out)


class PrecipitationEnhancementCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = max(1, out_channels // 4)
        total_branch_channels = branch_channels * 4
      
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, branch_channels, 3, padding=1), nn.BatchNorm2d(branch_channels), nn.ReLU(True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, branch_channels, 5, padding=2), nn.BatchNorm2d(branch_channels), nn.ReLU(True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, branch_channels, 7, padding=3), nn.BatchNorm2d(branch_channels), nn.ReLU(True))
        self.branch4 = nn.Sequential(nn.Conv2d(in_channels, branch_channels, 3, padding=2, dilation=2), nn.BatchNorm2d(branch_channels), nn.ReLU(True))

        self.fusion = nn.Sequential(
            nn.Conv2d(total_branch_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        concat = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return self.fusion(concat) + self.residual(x)

class CrossAttentionMamba(nn.Module):
    def __init__(self, img_size=66, img_size_w=70, patch_size=2, stride=2, in_chans=1, num_classes=3, dim=128, depth=4, d_state=64, dropout=0.15):
        super().__init__()
      
        self.num_classes = num_classes 
      
        self.patch_size = patch_size
        self.stride = stride
        self.dim = dim
        self.img_size = img_size
        self.img_size_w = img_size_w
      
        self.num_patches_h = (img_size - patch_size) // stride + 1
        self.num_patches_w = (img_size_w - patch_size) // stride + 1
        num_patches = self.num_patches_h * self.num_patches_w

        # Patch Embedding
        def build_patch_embed():
            return nn.Sequential(
                nn.Conv2d(in_chans, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(dim // 2),
                nn.GELU(),
                nn.Conv2d(dim // 2, dim, kernel_size=patch_size, stride=stride)
            )
      
        self.patch_embed1 = build_patch_embed()
        self.patch_embed2 = build_patch_embed()
      
        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, dim))
      
        self.blocks1 = nn.ModuleList([OmniBiMambaBlock_Pseudo(dim=dim, d_state=d_state, expand=2, dropout=dropout) for _ in range(depth)])
        self.blocks2 = nn.ModuleList([OmniBiMambaBlock_Pseudo(dim=dim, d_state=d_state, expand=2, dropout=dropout) for _ in range(depth)])
      
        self.cross_attn = GatedCrossAttentionBlock(dim, num_heads=8, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
      
        self.head = nn.Linear(dim, patch_size * patch_size * num_classes)
        self.cnn_enhancement = PrecipitationEnhancementCNN(num_classes, num_classes)

    def forward(self, img1, img2):
        x1 = self.patch_embed1(img1).flatten(2).transpose(1, 2)
        x2 = self.patch_embed2(img2).flatten(2).transpose(1, 2)
      
        x1 = x1 + self.pos_embed1
        x2 = x2 + self.pos_embed2

        current_H, current_W = self.num_patches_h, self.num_patches_w
      
        for blk in self.blocks1: x1 = blk(x1,current_H,current_W)
        for blk in self.blocks2: x2 = blk(x2,current_H,current_W)
      
        x_fuse = self.cross_attn(x1, x2)
        x_fuse = self.norm(x_fuse)
        x_fuse = self.head(x_fuse)
      
        B = x_fuse.size(0)
        # 使用 self.num_classes 进行 reshape
        x_fuse = x_fuse.view(B, self.num_patches_h, self.num_patches_w, self.patch_size, self.patch_size, self.num_classes)
        x_fuse = x_fuse.permute(0, 5, 1, 3, 2, 4).contiguous()
        x_fuse = x_fuse.view(B, self.num_classes, self.num_patches_h * self.patch_size, self.num_patches_w * self.patch_size)
      
        x_raw = torch.nn.functional.interpolate(x_fuse, size=(self.img_size, self.img_size_w), mode='bilinear', align_corners=False)
        x_final = self.cnn_enhancement(x_raw)
      
        return x_final

# ==========================================
# 2. 数据处理与工具
# ==========================================

class TripleChannelDataset(Dataset):
    def __init__(self, folder1_paths, folder2_paths, target_paths_1h, target_paths_2h, target_paths_3h, transform=None):
        self.folder1_paths = folder1_paths
        self.folder2_paths = folder2_paths
        self.targets = list(zip(target_paths_1h, target_paths_2h, target_paths_3h))
        self.transform = transform

    def __len__(self):
        return len(self.folder1_paths)

    def __getitem__(self, idx):
        img1 = Image.open(self.folder1_paths[idx]).convert('L')
        img2 = Image.open(self.folder2_paths[idx]).convert('L')
        t1, t2, t3 = self.targets[idx]
        target_1h = Image.open(t1).convert('L')
        target_2h = Image.open(t2).convert('L')
        target_3h = Image.open(t3).convert('L')
      
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            target_1h = self.transform(target_1h)
            target_2h = self.transform(target_2h)
            target_3h = self.transform(target_3h)
          
        targets = torch.stack([target_1h.squeeze(), target_2h.squeeze(), target_3h.squeeze()], dim=0)
        return img1, img2, targets

def plot_losses(train, val):
    plt.figure(figsize=(8, 5))
    plt.plot(train, label='Train Loss')
    plt.plot(val, label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_FIGURES_DIR, 'loss_curve.png'))
    plt.close()

def plot_gate_history(history):
    # 处理 None 值
    processed = []
    last = 0.5
    for h in history:
        if h is not None: last = h
        processed.append(last)
  
    plt.figure(figsize=(8, 4))
    plt.plot(processed)
    plt.title('Gate Value Evolution')
    plt.savefig(os.path.join(SAVE_FIGURES_DIR, 'gate_curve.png'))
    plt.close()

def show_results(model, dataloader, device, epoch, num_samples=3):
    """8 列对比图"""
    model.eval()
  
    # 1. 抓取一个 Batch
    try:
        img1, img2, targets = next(iter(dataloader))
    except StopIteration:
        return

    img1, img2 = img1.to(device), img2.to(device)
    with torch.no_grad():
        preds = model(img1, img2) # [B, 3, H, W]
  
    # 转 CPU
    img1 = img1.cpu()
    img2 = img2.cpu()
    preds = preds.cpu().clamp(0, 1)
    targets = targets.cpu()
  
    # 2. 绘图
    fig, axes = plt.subplots(num_samples, 8, figsize=(20, 3*num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)
  
    titles = ["Input PWV", "Input Radar", 
              "Pred +1h", "Target +1h", 
              "Pred +2h", "Target +2h", 
              "Pred +3h", "Target +3h"]
  
    for i in range(min(num_samples, img1.size(0))):
        # Input 1
        axes[i,0].imshow(img1[i].squeeze(), cmap='gray')
        # Input 2
        axes[i,1].imshow(img2[i].squeeze(), cmap='jet')
      
        # Pairs
        for t in range(3):
            # Pred
            axes[i, 2 + t*2].imshow(preds[i, t], cmap='jet', vmin=0, vmax=1)
            # Target
            axes[i, 3 + t*2].imshow(targets[i, t], cmap='jet', vmin=0, vmax=1)
          
        if i == 0:
            for col in range(8): axes[i, col].set_title(titles[col])
          
        for col in range(8): axes[i, col].axis('off')
          
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_FIGURES_DIR, f'vis_epoch_{epoch+1}.png'))
    plt.close()

# ==========================================
# 补充: 频域增强的连续物理加权损失函数
# ==========================================
class SpectralStructuralWeightedLoss(nn.Module):
    def __init__(self, w_mae=1.0,w_fft=0.1,w_ssim=0.2,heavy_rain_boost=6.0):
        '''
        __init__ 的 Docstring
        
        :param w_mae: 基础回归损失的权重
        :param w_fft: 频域损失权重(抗模糊核心)
        :param w_ssim: 结构相似性权重
        :param heavy_rain_boost: 强降水权重的动态系数
        '''
        super().__init__()
        self.w_mae = w_mae
        self.w_fft = w_fft
        self.w_ssim = w_ssim
        self.heavy_rain_boost = heavy_rain_boost
    
    def continuous_weight_l1(self,pred, target):
        diff = torch.abs(pred - target)
        weights = 1.0 + (target * self.heavy_rain_boost)
        return (weights * diff).mean()
    
    def fft_loss(self, pred, target):
        pred_fft = torch.fft.rfft2(pred,norm = 'ortho')
        target_fft = torch.fft.rfft2(target,norm = 'ortho')
        loss_fft = torch.mean(torch.abs(pred_fft - target_fft))
        return loss_fft
    
    def ssim_loss_simple(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(pred, 3, 1, 1)
        mu2 = F.avg_pool2d(target, 3, 1, 1)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1-ssim_map.mean()
    
    def forward(self, pred, target):
        loss_mae = self.continuous_weight_l1(pred, target)
        loss_fft = self.fft_loss(pred, target)
        B,T,H,W = pred.shape
        pred_flat = pred.view(-1,1,H,W)
        target_flat = target.view(-1,1,H,W)
        loss_ssim = self.ssim_loss_simple(pred_flat, target_flat)

        total_loss = self.w_mae * loss_mae + self.w_fft * loss_fft + self.w_ssim * loss_ssim
        return total_loss

# ==========================================
# 3. 验证与训练循环
# ==========================================

def validate(model, dataloader, device, verbose=True):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
  
    # 累加器
    total_mae = [0]*3
    total_mape = [0]*3
    total_psnr = [0]*3
    total_ssim = [0]*3
    num_batches = 0
  
    with torch.no_grad():
        for img1, img2, targets in dataloader:
            img1, img2, targets = img1.to(device), img2.to(device), targets.to(device)
            output = model(img1, img2)
          
            loss = criterion(output, targets)
            total_loss += loss.item()
          
            # 详细指标计算
            for t in range(3):
                out_t = output[:, t]
                tgt_t = targets[:, t]
              
                # MAE
                total_mae[t] += torch.mean(torch.abs(out_t - tgt_t)).item()
              
                # MAPE
                total_mape[t] += (torch.mean(torch.abs((out_t - tgt_t) / (tgt_t + 1e-8))) * 100).item()
              
                # PSNR
                mse_val = torch.mean((out_t - tgt_t) ** 2)
                if mse_val > 0:
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_val))
                    total_psnr[t] += psnr.item()
                else:
                    total_psnr[t] += 100
              
                # Manual SSIM (Copied from sota.py to avoid dependency issues)
                mu1 = torch.mean(out_t)
                mu2 = torch.mean(tgt_t)
                sigma1_sq = torch.var(out_t)
                sigma2_sq = torch.var(tgt_t)
                sigma12 = torch.mean((out_t - mu1) * (tgt_t - mu2))
                c1, c2 = 0.01**2, 0.03**2
                ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
                total_ssim[t] += ssim_val.item()
              
            num_batches += 1
          
    avg_loss = total_loss / num_batches
    if verbose:
        print(f"\n[Validation] MSE: {avg_loss:.5f}")
        for t in range(3):
            print(f"  T+{t+1}h | MAE: {total_mae[t]/num_batches:.4f} | MAPE: {total_mape[t]/num_batches:.2f}% | PSNR: {total_psnr[t]/num_batches:.2f} | SSIM: {total_ssim[t]/num_batches:.4f}")
          
    return avg_loss

def train(model, tr_loader, val_loader, device, epochs):
    #criterion = nn.MSELoss()
    # 应用修改的损失函数，可以根据训练情况调整w_fft的值
    criterion = SpectralStructuralWeightedLoss(w_mae=1.0,w_fft=0.05,w_ssim=0.1,heavy_rain_boost=5.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
  
    train_losses = []
    val_losses = []
    gate_history = []
    best_val_loss = float('inf')
    patience = 35
    counter = 0
  
    print(f"Start training for {epochs} epochs...")
  
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        gate_accum = 0
        gate_count = 0
      
        for img1, img2, targets in tr_loader:
            img1, img2, targets = img1.to(device), img2.to(device), targets.to(device)
          
            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
          
            running_loss += loss.item()
          
            if model.cross_attn.last_gate_mean is not None:
                gate_accum += model.cross_attn.last_gate_mean
                gate_count += 1
      
        scheduler.step()
      
        avg_trn_loss = running_loss / len(tr_loader)
        avg_gate = gate_accum / gate_count if gate_count > 0 else None
      
        train_losses.append(avg_trn_loss)
        gate_history.append(avg_gate)
      
        # 验证
        val_loss = validate(model, val_loader, device, verbose=False)
        val_losses.append(val_loss)
      
        print(f"Epoch {epoch+1}/{epochs} | TrnLoss: {avg_trn_loss:.5f} | ValLoss: {val_loss:.5f} | Gate: {avg_gate if avg_gate else 'N/A'}")
      
        # 保存最佳
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_FIGURES_DIR, "best_model.pth"))
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
      
        # 可视化 (每5轮)
        if (epoch + 1) % 5 == 0:
            show_results(model, val_loader, device, epoch)
            # 同时打印一次详细指标
            validate(model, val_loader, device, verbose=True)

    # 结束
    plot_losses(train_losses, val_losses)
    plot_gate_history(gate_history)
    torch.save(model.state_dict(), os.path.join(SAVE_FIGURES_DIR, "final_model.pth"))

# ==========================================
# 4. 主程序
# ==========================================

if __name__ == "__main__":
    # 路径
    folder1 = "./sota_data/PWV"
    folder2 = "./sota_data/RADAR"
    target_folder = "./sota_data/RAIN"
  
    # 检查
    if not all(os.path.exists(p) for p in [folder1, folder2, target_folder]):
        print("Error: Data folders not found. Please ensure ./sota_data/ exists.")
        exit(1)

    # 1. 匹配文件 (Copy from sota.py)
    def get_dict(f): return {os.path.splitext(n)[0]: os.path.join(f, n) for n in os.listdir(f)}
    f1_dict = get_dict(folder1)
    f2_dict = get_dict(folder2)
    tg_dict = get_dict(target_folder)
  
    def parse_time(name):
        for fmt in ["%Y-%m-%d-%H-%M-%S", "%Y-%m-%d-%H-%M"]:
            try: return datetime.strptime(name, fmt)
            except: continue
        return None
      
    matched_keys = []
    print("Matching files...")
  
    for k1, p1 in f1_dict.items():
        t1 = parse_time(k1)
        if not t1: continue
      
        # 找最近的 Radar (1小时内)
        best_k2, min_diff = None, float('inf')
        for k2 in f2_dict:
            t2 = parse_time(k2)
            if t2:
                diff = abs((t1 - t2).total_seconds())
                if diff < min_diff: min_diff, best_k2 = diff, k2
      
        if best_k2 and min_diff <= 3600:
            # 找未来3小时 Target
            targets = []
            valid = True
            for h in range(1, 4):
                tk = (t1 + timedelta(hours=h)).strftime("%Y-%m-%d-%H-%M-%S")
                if tk in tg_dict: targets.append(tg_dict[tk])
                else: valid = False; break
          
            if valid:
                matched_keys.append((p1, f2_dict[best_k2], *targets))
              
    print(f"Found {len(matched_keys)} matched samples.")
    if len(matched_keys) == 0: exit(1)
  
    # 2. 划分数据集
    data_list = list(zip(*matched_keys)) # 解压
    # split: f1_tr, f1_val, f2_tr, f2_val...
    split_res = train_test_split(*data_list, test_size=0.2, random_state=42)
  
    tr_f1, val_f1 = split_res[0], split_res[1]
    tr_f2, val_f2 = split_res[2], split_res[3]
    tr_t1, val_t1 = split_res[4], split_res[5]
    tr_t2, val_t2 = split_res[6], split_res[7]
    tr_t3, val_t3 = split_res[8], split_res[9]
  
    # 3. DataLoader
    tf = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size_w'])),
        transforms.ToTensor(),
    ])
  
    tr_ds = TripleChannelDataset(tr_f1, tr_f2, tr_t1, tr_t2, tr_t3, transform=tf)
    val_ds = TripleChannelDataset(val_f1, val_f2, val_t1, val_t2, val_t3, transform=tf)
  
    tr_loader = DataLoader(tr_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
  
    # 4. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
  
    model = CrossAttentionMamba(
        img_size=CONFIG['img_size'],
        img_size_w=CONFIG['img_size_w'],
        patch_size=CONFIG['patch_size'],
        stride=CONFIG['stride'],
        d_state=CONFIG['d_state'],
        dim=CONFIG['dim'],
        depth=CONFIG['depth'],
        num_classes=3  
    ).to(device)
  
    # 5. 运行
    train(model, tr_loader, val_loader, device, CONFIG['epochs'])
