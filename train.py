import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from Model.GeneratorV2 import Generator
from Model.Discriminator import Discriminator

# Minimal placeholders - replace with your actual implementations
class SRDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform or transforms.ToTensor()
        self.paths = []
        for dp, _, fns in os.walk(root):
            for f in fns:
                if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                    self.paths.append(os.path.join(dp, f))
        assert len(self.paths) > 0, f"No images found in {root}"

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("L")
        hr = self.transform(img)
        lr = torch.nn.functional.interpolate(hr.unsqueeze(0), scale_factor=0.25, mode='bicubic', align_corners=False).squeeze(0)
        return lr, hr

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=35):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        self.features = vgg19(weights=VGG19_Weights.DEFAULT).features[:layer_index+1].eval()
        for p in self.features.parameters():
            p.requires_grad = False
    def forward(self, x): return self.features(x)

def main():
    import os
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda")
data_path = "C:\\Users\\SOEE\\Documents\\GitHub\\W-GAN-DJ\\datasets\\CT-dataset"
transform = transforms.ToTensor()
dataset = SRDataset(data_path, transform)

# Optimized DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=16,  # Increased from 16 for better GPU utilization
    shuffle=True, 
    pin_memory=True 
  # Reduces worker restart overhead
)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
vgg = VGGFeatureExtractor(layer_index=22).to(device)
vgg.eval()

# Changed to BCEWithLogitsLoss for mixed precision compatibility
criterion_gan = nn.BCEWithLogitsLoss()
criterion_content = nn.MSELoss()

opt_G = optim.Adam(generator.parameters(), lr=1e-3)
opt_D = optim.Adam(discriminator.parameters(), lr=1e-4)

# Mixed precision scalers
scaler_G = GradScaler()
scaler_D = GradScaler()

os.makedirs("sr_output-128_512v2", exist_ok=True)

epoch_losses = []

for epoch in range(2):
    total_loss = 0.0
    num_batches = 0

    for i, (lr, hr) in enumerate(dataloader):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        
        # Create targets directly on GPU
        bsz = lr.size(0)
        valid = torch.ones((bsz,), device=device)  # Removed extra dimension for BCEWithLogitsLoss
        fake = torch.zeros((bsz,), device=device)

        # Train Generator with mixed precision
        opt_G.zero_grad(set_to_none=True)
        with autocast():
            gen_hr = generator(lr)
            pred_fake = discriminator(gen_hr)
            loss_gan = criterion_gan(pred_fake, valid)

            # Use expand instead of repeat to avoid memory copies
            gen_hr_rgb = gen_hr.expand(-1, 3, -1, -1)
            hr_rgb = hr.expand(-1, 3, -1, -1)
            loss_content = criterion_content(vgg(gen_hr_rgb), vgg(hr_rgb))
            loss_G = loss_content + 1e-3 * loss_gan

        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()

        # Train Discriminator with mixed precision
        opt_D.zero_grad(set_to_none=True)
        with autocast():
            loss_real = criterion_gan(discriminator(hr), valid)
            loss_fake = criterion_gan(discriminator(gen_hr.detach()), fake)
            loss_D = 0.5 * (loss_real + loss_fake)

        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        
        print(f"[Epoch {epoch}] [Batch {i}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        last_gen_hr = gen_hr.detach()
        total_loss += loss_G.item()
        num_batches += 1

    avg_epoch_loss = total_loss / num_batches
    epoch_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch} Average Generator Loss: {avg_epoch_loss:.4f}")


if __name__ == "__main__":
    main()
