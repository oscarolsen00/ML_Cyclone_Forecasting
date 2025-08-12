import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset, DataLoader

# ==============================================
# 1) Load datasets (whole-domain segmentation)
# ==============================================
train_ds = xr.open_zarr('train_data_normalised.zarr', consolidated=True)
val_ds   = xr.open_zarr('val_data_normalised.zarr', consolidated=True)

train_inputs = train_ds['inputs'].data  # (N, H, W, C)
train_labels = train_ds['labels'].data  # (N, H, W)
val_inputs   = val_ds['inputs'].data
val_labels   = val_ds['labels'].data

# Preload val set if it fits in memory
val_inputs_np = val_inputs.compute()
val_labels_np = val_labels.compute()
val_inputs_tensor = torch.from_numpy(val_inputs_np).permute(0, 3, 1, 2).float()
val_labels_tensor = torch.from_numpy(val_labels_np).unsqueeze(1).float()

# ==============================================
# 2) Dataset
# ==============================================
class ERA5SegmentationDataset(Dataset):
    def __init__(self, inputs_da, labels_da):
        self.inputs = inputs_da
        self.labels = labels_da

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        img_np = self.inputs[idx].compute()   # (H, W, C)
        mask_np = self.labels[idx].compute()  # (H, W)
        img = torch.from_numpy(img_np).permute(2, 0, 1).float()  # (C,H,W)
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()    # (1,H,W)
        return img, mask

# ==============================================
# 3) DataLoaders
# ==============================================
batch_size = 4
train_loader = DataLoader(ERA5SegmentationDataset(train_inputs, train_labels),
                          batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(ERA5SegmentationDataset(val_inputs, val_labels),
                          batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

# ==============================================
# 4) U-Net model
# ==============================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, base_ch=64):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*8)
        self.up1 = Up(base_ch*16, base_ch*4)
        self.up2 = Up(base_ch*8,  base_ch*2)
        self.up3 = Up(base_ch*4,  base_ch)
        self.up4 = Up(base_ch*2,  base_ch)
        self.outc = OutConv(base_ch, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)  # (B,1,H,W)

# ==============================================
# 5) Training loop
# ==============================================
def dice_coeff(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2*inter + eps) / (union + eps)).mean().item()

def train_unet(model, train_loader, val_inputs_t=None, val_labels_t=None,
               epochs=20, lr=1e-4, weight_decay=1e-4, pos_weight_val=5.0):
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        if val_inputs_t is not None:
            model.eval()
            with torch.no_grad():
                v_logits = model(val_inputs_t.to(device))
                v_loss = criterion(v_logits, val_labels_t.to(device)).item()
                v_dice = dice_coeff(torch.sigmoid(v_logits), val_labels_t.to(device))
            print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {v_loss:.4f} | Dice: {v_dice:.4f}")
        else:
            print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}")

    return model

# ==============================================
# 6) Run training
# ==============================================
input_channels = train_inputs.shape[-1]
model = UNet(n_channels=input_channels, n_classes=1, base_ch=64)

trained_model = train_unet(model, train_loader,
                           val_inputs_t=val_inputs_tensor,
                           val_labels_t=val_labels_tensor,
                           epochs=20, lr=1e-4, weight_decay=1e-4,
                           pos_weight_val=5.0)

torch.save(trained_model.state_dict(), 'unet.pth')
print("Model saved as unet.pth")