import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dask.array as darray


# Load the datasets lazily
train_ds = xr.open_zarr('train_data_normalised.zarr', consolidated=True)
val_ds = xr.open_zarr('val_data_normalised.zarr', consolidated=True)

# Extract Dask arrays
train_inputs = train_ds['inputs'].data
train_centers = train_ds['centers'].data
train_labels = train_ds['labels'].data

val_inputs = val_ds['inputs'].data
val_centers = val_ds['centers'].data
val_labels = val_ds['labels'].data

# Load all val data into memory (eagerly compute once)
val_inputs_array = val_inputs.compute()
val_centers_array = val_centers.compute()
val_labels_array = val_labels.compute()

# Convert to torch tensors
val_inputs_tensor = torch.from_numpy(val_inputs_array).permute(0, 3, 1, 2).float()
val_coords_tensor = torch.from_numpy(val_centers_array).float()
val_labels_tensor = torch.from_numpy(val_labels_array).float()




# -----------------------------------------------------------------------------
# 2) Dataset that computes each sample on demand
# -----------------------------------------------------------------------------
class ERA5GridDataset(Dataset):
    def __init__(self, inputs, centers, labels):
        """
        inputs:  dask array of shape (N, H, W, C)
        centers: dask array of shape (N, 2)
        labels:  dask array of shape (N,)
        """
        self.inputs  = inputs
        self.centers = centers
        self.labels  = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        # pull just one sample into memory
        img_np   = self.inputs[i].compute()         # (H, W, C)
        coord_np = self.centers[i].compute()        # (2,)
        lab_val  = float(self.labels[i].compute())  # scalar

        # convert to torch
        img   = torch.from_numpy(img_np).permute(2, 0, 1).float()  # (C, H, W)
        coord = torch.from_numpy(coord_np).float()                 # (2,)
        lab   = torch.tensor(lab_val).float()                      # ()

        return img, coord, lab

# -----------------------------------------------------------------------------
# 3) Create DataLoaders
# -----------------------------------------------------------------------------
batch_size = 16


train_ds = ERA5GridDataset(train_inputs, train_centers, train_labels)
val_ds   = ERA5GridDataset(val_inputs,   val_centers,   val_labels)


train_loader = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,)

val_loader   = DataLoader(val_ds,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0,
                          )


print("DataLoaders ready:",
      f"train={len(train_ds)} samples,",
      f"val={len(val_ds)},")

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, z):
        b, c, _, _ = z.size()
        y = self.avg_pool(z).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return z * y.expand_as(z)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None, stride=1, reduction=4):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.silu = nn.SiLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample

    def forward(self, z):
        identity = z

        z = self.conv1(z)
        z = self.bn1(z)
        z = self.silu(z)
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.silu(z)
        z = self.conv3(z)
        z = self.bn3(z)
        z = self.se(z)

        if self.downsample is not None:
            identity = self.downsample(identity)

        z += identity
        z = self.silu(z)

        return z
    

class SENetWithCoords(nn.Module):
    def __init__(self, block, layers, image_channels, coord_dim=2):
        super(SENetWithCoords, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.silu = nn.SiLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], mid_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], mid_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], mid_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], mid_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Adding two more dimensions for the coordinates
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 + coord_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z, coords):
        """
        z: Tensor of shape (batch, input_channels, H, W)
        coords: Tensor of shape (batch, 2) giving (lat_center, lon_center)
        """
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.silu(z)
        z = self.maxpool(z)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.avgpool(z)
        z = z.view(z.size(0), -1)

        # Concatenate coordinates
        z = torch.cat([z, coords], dim=1)  # [batch, F + 2]
        z = self.fc(z)
        return z

    def _make_layer(self, block, num_residual_blocks, mid_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != mid_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, mid_channels * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * 4)
            )

        layers.append(block(self.in_channels, mid_channels,
                            identity_downsample, stride))
        self.in_channels = mid_channels * 4
        
        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, mid_channels))

        return nn.Sequential(*layers)


def SDS_SENetWithCoords(img_channel):
    return SENetWithCoords(SEBottleneck, [3, 4, 6, 3], img_channel)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def train_model(model, train_loader, val_inputs=None, val_coords=None, val_labels=None,
                       epochs=30, learning_rate=1e-4, weight_decay=1e-3):
    """
    Trains a ResNet model whose forward signature is model(inputs, coords).
    Also, returns training and validation losses for each epoch.
    """
    # 1) pick device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)

    # 2) loss & optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # ---- training pass ----
        model.train()
        running_loss = 0.0

        for inputs, coords, labels in train_loader:
            inputs, coords, labels = (
                inputs.to(device),
                coords.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs, coords).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- validation pass ----
        if val_inputs is not None:
            model.eval()
            with torch.no_grad():
                outputs = model(val_inputs.to(device), val_coords.to(device)).squeeze()
                loss = criterion(outputs, val_labels.to(device))
                val_losses.append(loss.item())
            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss:   {loss:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f}")

    print("Training complete.")
    return model, train_losses, val_losses

# ---------------------------
# Initialize Device
# ---------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# ---------------------------
# Initialize Model
# ---------------------------
input_channels = train_inputs.shape[-1]  # this should be the number of channels in your ERA5 data
model = SDS_SENetWithCoords(input_channels)
model.apply(init_weights)
model.to(device)
print("Model initialized and moved to device.")

# ---------------------------
# Define Hyperparameters
# ---------------------------
learning_rate = 1e-4
weight_decay = 1e-4
epochs = 30

# ---------------------------
# Train the Model
# ---------------------------
trained_model, train_losses, val_losses = train_model(
    model,
    train_loader,
    val_loader,
    epochs=epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay
)

# ---------------------------
# Save the Model
# ---------------------------
model_name = "SENet.pth"
torch.save(trained_model.state_dict(), model_name)
print(f"Model saved as '{model_name}'")




