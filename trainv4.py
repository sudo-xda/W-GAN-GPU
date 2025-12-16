import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import torch._dynamo
torch._dynamo.config.suppress_errors = True

#from model.model_w import Generator, Discriminator
#from model.new_w_model import Generator, Discriminator
#from model.srgan_model import Generator, Discriminator
from model.SRCNN_model import SRCNN


class SRDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform or transforms.ToTensor()
        self.paths = []
        for dp, _, fns in os.walk(root):
            for f in fns:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    self.paths.append(os.path.join(dp, f))
        assert len(self.paths) > 0, f"No images found in {root}"

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("L")
        hr = self.transform(img)
        lr = torch.nn.functional.interpolate(
            hr.unsqueeze(0), scale_factor=0.25, mode='bicubic', align_corners=False
        ).squeeze(0)
        return lr, hr


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=22):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        self.features = vgg19(weights=VGG19_Weights.DEFAULT).features[:layer_index+1].eval()
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x): 
        return self.features(x)


def main():
    
    run_name = "srgan_run-psnr-MRI_100epoch_SRCNN"   
    device = torch.device("cuda")
    data_path = "C:\\Users\\SOEE\\Documents\\GitHub\\W-GAN-GPU\\dataset\\MRI_kaggle_512"

    # Folders
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    output_dir = os.path.join(run_dir, "images")
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # CSV log
    log_file = os.path.join(run_dir, f"{run_name}_log.csv")
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'D_loss', 'G_loss', 'content_loss', 'gan_loss', 'PSNR', 'SSIM'])

    transform = transforms.ToTensor()
    dataset = SRDataset(data_path, transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    vgg = VGGFeatureExtractor(layer_index=22).to(device).eval()

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_content = nn.MSELoss()
    #criterion_content = nn.L1Loss()

    opt_G = optim.Adam(generator.parameters(), lr=1e-3)
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-5)

    # Updated AMP scaler syntax
    scaler_G = torch.amp.GradScaler("cuda")
    scaler_D = torch.amp.GradScaler("cuda")

    # Metrics (specify data_range since images are normalized 0–1)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


    total_epochs = 100

    for epoch in tqdm(range(total_epochs), desc="Epochs", unit="epoch"):
        total_G_loss, total_D_loss, total_content_loss, total_gan_loss = 0, 0, 0, 0
        total_psnr, total_ssim = 0, 0
        num_batches = 0

        batch_pbar = tqdm(enumerate(dataloader),
                         total=len(dataloader),
                         desc=f"Epoch {epoch+1}/{total_epochs}",
                         leave=False,
                         unit="batch")

        for i, (lr, hr) in batch_pbar:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            bsz = lr.size(0)
            valid = torch.ones((bsz,), device=device)
            fake = torch.zeros((bsz,), device=device)

            # Train Generator
            opt_G.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                gen_hr = generator(lr)
                gen_hr = torch.clamp(gen_hr, 0.0, 1.0)
                pred_fake = discriminator(gen_hr)
                loss_gan = criterion_gan(pred_fake, valid)

                gen_hr_rgb = gen_hr.expand(-1, 3, -1, -1)
                hr_rgb = hr.expand(-1, 3, -1, -1)
                loss_content = criterion_content(vgg(gen_hr_rgb), vgg(hr_rgb))
                loss_G = loss_content + 1e-3 * loss_gan

            scaler_G.scale(loss_G).backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler_G.step(opt_G)
            scaler_G.update()

            # Train Discriminator
            opt_D.zero_grad(set_to_none=True)
            with autocast():
                loss_real = criterion_gan(discriminator(hr), valid)
                loss_fake = criterion_gan(discriminator(gen_hr.detach()), fake)
                loss_D = 0.5 * (loss_real + loss_fake)
               
            scaler_D.scale(loss_D).backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            scaler_D.step(opt_D)
            scaler_D.update()

            # Metrics
            psnr_val = psnr_metric(gen_hr, hr).item()
            ssim_val = ssim_metric(gen_hr, hr).item()

            batch_pbar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}',
                'PSNR': f'{psnr_val:.2f}',
                'SSIM': f'{ssim_val:.3f}'
            })

            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()
            total_content_loss += loss_content.item()
            total_gan_loss += loss_gan.item()
            total_psnr += psnr_val
            total_ssim += ssim_val
            num_batches += 1

            last_gen_hr = gen_hr.detach()

        batch_pbar.close()

        save_image(last_gen_hr, os.path.join(output_dir, f"epoch_{epoch:03d}.png"))

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': opt_G.state_dict(),
                'optimizer_D_state_dict': opt_D.state_dict(),
                'scaler_G_state_dict': scaler_G.state_dict(),
                'scaler_D_state_dict': scaler_D.state_dict(),
            }, os.path.join(weights_dir, f'checkpoint_epoch_{epoch+1:03d}.pth'))
            print(f"\n✅ Saved checkpoint at epoch {epoch+1}")

        # Average metrics
        avg_G_loss = total_G_loss / num_batches
        avg_D_loss = total_D_loss / num_batches
        avg_content_loss = total_content_loss / num_batches
        avg_gan_loss = total_gan_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_D_loss, avg_G_loss, avg_content_loss, avg_gan_loss, avg_psnr, avg_ssim])

        print(f"Epoch {epoch+1} ✅ Avg G Loss: {avg_G_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.3f}")

    print(f"\n Training Complete!")
    print(f" Training log saved to: {log_file}")
    print(f" Images saved to: {output_dir}")
    print(f" Weights saved to: {weights_dir}")


if __name__ == "__main__":
    main()
