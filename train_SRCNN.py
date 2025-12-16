import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from model.SRCNN_model import SRCNN
from dataset.SRDataset import SRDataset


def save_sample_images(epoch, lr_batch, hr_batch, sr_batch, psnr_metric, output_dir, paths):
    for idx in range(min(5, lr_batch.size(0))):
        hr_img = hr_batch[idx].cpu()
        lr_img = lr_batch[idx].cpu()

        # upsample LR just for display
        import torch.nn.functional as F
        lr_up = F.interpolate(lr_img.unsqueeze(0), size=hr_img.shape[1:], mode="bicubic").squeeze(0)

        sr_img = sr_batch[idx].cpu()

        psnr_val = psnr_metric(sr_img.unsqueeze(0), hr_img.unsqueeze(0)).item()

        combined = torch.cat([hr_img, lr_up, sr_img], dim=-1)

        base = os.path.splitext(os.path.basename(paths[idx]))[0]
        fname = f"epoch_{epoch:03d}_{base}_PSNR{psnr_val:.2f}.png"
        save_image(combined, os.path.join(output_dir, fname))


def main():

    run_name = "SRCNN_RUN_CT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "dataset//CT-SMALL-512"

    # folders
    run_dir = os.path.join("runs", run_name)
    output_dir = os.path.join(run_dir, "images")
    weights_dir = os.path.join(run_dir, "weights")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # CSV log
    log_file = os.path.join(run_dir, f"{run_name}.csv")
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss", "PSNR", "SSIM"])

    # dataset
    dataset = SRDataset(data_path, scale=2)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # model
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    epochs = 50

    for epoch in range(epochs):
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        n = 0

        for i, (lr, hr, paths) in enumerate(tqdm(loader)):
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)

            loss = criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = psnr_metric(sr, hr).item()
            ssim = ssim_metric(sr, hr).item()

            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim
            n += 1

            if i == 0:
                save_sample_images(epoch+1, lr, hr, sr, psnr_metric, output_dir, paths)

        avg_loss = total_loss / n
        avg_psnr = total_psnr / n
        avg_ssim = total_ssim / n

        # write log
        with open(log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_loss, avg_psnr, avg_ssim])

        torch.save(model.state_dict(), os.path.join(weights_dir, f"SRCNN_epoch_{epoch+1}.pth"))

        print(f"Epoch {epoch+1}/{epochs}  Loss={avg_loss:.4f}  PSNR={avg_psnr:.2f}  SSIM={avg_ssim:.3f}")

    print("\nTraining Done!")


if __name__ == "__main__":
    main()
