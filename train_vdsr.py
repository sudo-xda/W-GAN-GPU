import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# --------------------------------------------------
# Dataset (HR only → auto LR generation)
# --------------------------------------------------
class MedicalSRDataset(Dataset):
    def __init__(self, root, scale=4):
        self.scale = scale
        self.paths = []
        for dp, _, fns in os.walk(root):
            for f in fns:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    self.paths.append(os.path.join(dp, f))

        assert len(self.paths) > 0, "No images found!"
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        hr = self.transform(img)

        lr = F.interpolate(
            hr.unsqueeze(0),
            scale_factor=1 / self.scale,
            mode="bicubic",
            align_corners=False
        ).squeeze(0)

        lr_up = F.interpolate(
            lr.unsqueeze(0),
            size=hr.shape[-2:],
            mode="bicubic",
            align_corners=False
        ).squeeze(0)

        return lr_up, hr


# --------------------------------------------------
# VDSR Model
# --------------------------------------------------
class VDSR(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(1, 64, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(18):
            layers.append(nn.Conv2d(64, 64, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)  # residual learning


# --------------------------------------------------
# Training Function
# --------------------------------------------------
def train_vdsr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = r"C:\\Users\SOEE\\Documents\\GitHub\\W-GAN-GPU\\dataset\\CT-SMALL-512"
    save_dir = "runs/vdsr_medical"
    os.makedirs(save_dir, exist_ok=True)

    dataset = MedicalSRDataset(data_path, scale=4)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True)

    model = VDSR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    log_path = os.path.join(save_dir, "vdsr_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["Epoch", "Loss", "PSNR", "SSIM"])

    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss, total_psnr, total_ssim = 0, 0, 0

        for lr, hr in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_psnr += psnr(sr, hr).item()
            total_ssim += ssim(sr, hr).item()

        avg_loss = total_loss / len(loader)
        avg_psnr = total_psnr / len(loader)
        avg_ssim = total_ssim / len(loader)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, avg_loss, avg_psnr, avg_ssim])

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"vdsr_epoch_{epoch+1}.pth"))
            save_image(sr[:4], os.path.join(save_dir, f"sample_epoch_{epoch+1}.png"))

    print("✅ VDSR Training Complete")


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    train_vdsr()
