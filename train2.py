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

import torch._dynamo
torch._dynamo.config.suppress_errors = True

#from Model.m2 import Generator, Discriminator
from Model.GeneratorV2 import Generator
from Model.m2 import Discriminator

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

def save_training_images(lr, hr, sr, epoch, output_dir):
    """Save LR, HR, and SR images for comparison"""
    # Take first image from batch
    lr_img = lr[0:1]  # [1, 1, H, W]
    hr_img = hr[0:1]  # [1, 1, H, W] 
    sr_img = sr[0:1]  # [1, 1, H, W]
    
    # Create comparison grid
    comparison = torch.cat([lr_img, sr_img, hr_img], dim=3)  # Horizontal concat
    save_image(comparison, os.path.join(output_dir, f"epoch_{epoch:03d}_comparison.png"))
    
    # Save individual images
    save_image(lr_img, os.path.join(output_dir, f"epoch_{epoch:03d}_LR.png"))
    save_image(hr_img, os.path.join(output_dir, f"epoch_{epoch:03d}_HR.png"))
    save_image(sr_img, os.path.join(output_dir, f"epoch_{epoch:03d}_SR.png"))

def main():
    device = torch.device("cuda")
    data_path = "C:\\Users\\SOEE\\Documents\\GitHub\\W-GAN-DJ\\datasets\\CT-dataset"
    output_dir = "sr_output-enhancedv2"
    weights_dir = "weightsv2"
    log_file = "training_logv2.csv"
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    # Initialize CSV logging
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'batch', 'D_loss', 'G_loss', 'content_loss', 'gan_loss'])
    
    transform = transforms.ToTensor()
    dataset = SRDataset(data_path, transform)
    
    # Windows-friendly DataLoader (no num_workers)
    dataloader = DataLoader(
        dataset, 
        batch_size=16,
        shuffle=True, 
        pin_memory=True
    )
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    vgg = VGGFeatureExtractor(layer_index=22).to(device)
    vgg.eval()
    
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_content = nn.MSELoss()
    
    opt_G = optim.Adam(generator.parameters(), lr=1e-3)
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    
    scaler_G = GradScaler()
    scaler_D = GradScaler()
    
    epoch_losses = []
    total_epochs = 100  # Changed to 100 for weight saving demo
    
    # Main training loop with progress bar
    for epoch in tqdm(range(total_epochs), desc="Epochs", unit="epoch"):
        total_loss = 0.0
        num_batches = 0
        
        # Batch progress bar
        batch_pbar = tqdm(enumerate(dataloader), 
                         total=len(dataloader), 
                         desc=f"Epoch {epoch+1}/{total_epochs}", 
                         leave=False, 
                         unit="batch")
        
        for i, (lr, hr) in batch_pbar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            
            bsz = lr.size(0)
            valid = torch.ones((bsz,), device=device)
            fake = torch.zeros((bsz,), device=device)
            
            # Train Generator
            opt_G.zero_grad(set_to_none=True)
            with autocast():
                gen_hr = generator(lr)
                pred_fake = discriminator(gen_hr)
                loss_gan = criterion_gan(pred_fake, valid)
                
                gen_hr_rgb = gen_hr.expand(-1, 3, -1, -1)
                hr_rgb = hr.expand(-1, 3, -1, -1)
                loss_content = criterion_content(vgg(gen_hr_rgb), vgg(hr_rgb))
                loss_G = loss_content + 1e-3 * loss_gan
            
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
            
            # Train Discriminator
            opt_D.zero_grad(set_to_none=True)
            with autocast():
                loss_real = criterion_gan(discriminator(hr), valid)
                loss_fake = criterion_gan(discriminator(gen_hr.detach()), fake)
                loss_D = 0.5 * (loss_real + loss_fake)
            
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()
            
            # Update progress bar
            batch_pbar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}',
                'Content': f'{loss_content.item():.4f}'
            })
            
            # Log to CSV every batch
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, i, loss_D.item(), loss_G.item(), 
                               loss_content.item(), loss_gan.item()])
            
            #last_lr, last_hr, last_sr = lr.detach(), hr.detach(), gen_hr.detach()
            last_gen_hr = gen_hr.detach()
            total_loss += loss_G.item()
            num_batches += 1
        
        batch_pbar.close()
        
        save_image(last_gen_hr, f"/epoch_{epoch:03d}.png")
        # Save images at end of each epoch
        #save_training_images(last_lr, last_hr, last_sr, epoch, output_dir)
        
        # Save model weights every 10 epochs
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
        
        avg_epoch_loss = total_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1} Complete - Average G Loss: {avg_epoch_loss:.4f}")

    print(f"\n🎉 Training Complete!")
    print(f"📊 Training log saved to: {log_file}")
    print(f"🖼️  Images saved to: {output_dir}")
    print(f"💾 Weights saved to: {weights_dir}")

if __name__ == "__main__":
    main()
