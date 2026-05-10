import argparse
import glob
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from main import Discriminator, FlatFolderDataset, Generator, _dir_has_subdirs


@dataclass(frozen=True)
class LossRow:
    epoch: int
    loss_d: float
    loss_g: float
    d_x: float
    d_g_z1: float
    d_g_z2: float


_EPOCH_RE = re.compile(r"netG_epoch_(\d+)\.pth$")


def _infer_epochs(checkpoint_dir: str) -> list[int]:
    epochs: list[int] = []
    for path in glob.glob(os.path.join(checkpoint_dir, "netG_epoch_*.pth")):
        m = _EPOCH_RE.search(path)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(set(epochs))


def _build_real_batch(dataroot: str, image_size: int, batch_size: int, device: torch.device):
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if _dir_has_subdirs(dataroot):
        from torchvision.datasets import ImageFolder

        dataset = ImageFolder(root=dataroot, transform=tfm)
    else:
        dataset = FlatFolderDataset(root=dataroot, transform=tfm)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    real_cpu, _ = next(iter(loader))
    return real_cpu.to(device)


def _select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda:0")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {requested}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DCGAN checkpoints on a fixed batch")
    parser.add_argument("--dataroot", required=True, help="Path to image folder (CelebA-HQ-img)")
    parser.add_argument("--ckptdir", required=True, help="Directory containing netG_epoch_*.pth and netD_epoch_*.pth")
    parser.add_argument("--out", default=None, help="Output directory for CSV/plot (default: --ckptdir)")
    parser.add_argument("--device", default="auto", help="cpu | cuda | auto")
    parser.add_argument("--imageSize", type=int, default=64)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    out_dir = args.out or args.ckptdir
    os.makedirs(out_dir, exist_ok=True)

    device = _select_device(args.device)
    print(f"Using device: {device}")

    epochs = _infer_epochs(args.ckptdir)
    if not epochs:
        raise RuntimeError(f"No checkpoints found in: {args.ckptdir}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    real = _build_real_batch(args.dataroot, args.imageSize, args.batchSize, device)
    fixed_noise = torch.randn(args.batchSize, args.nz, 1, 1, device=device)

    criterion = nn.BCELoss()
    rows: list[LossRow] = []

    for epoch in epochs:
        g_path = os.path.join(args.ckptdir, f"netG_epoch_{epoch}.pth")
        d_path = os.path.join(args.ckptdir, f"netD_epoch_{epoch}.pth")
        if not (os.path.exists(g_path) and os.path.exists(d_path)):
            continue

        netG = Generator(nz=args.nz, ngf=args.ngf, nc=3, ngpu=1).to(device)
        netD = Discriminator(nc=3, ndf=args.ndf, ngpu=1).to(device)
        netG.load_state_dict(torch.load(g_path, map_location=device))
        netD.load_state_dict(torch.load(d_path, map_location=device))
        netG.eval()
        netD.eval()

        with torch.no_grad():
            label_real = torch.ones((real.size(0),), dtype=real.dtype, device=device)
            label_fake = torch.zeros((real.size(0),), dtype=real.dtype, device=device)

            out_real = netD(real)
            errD_real = criterion(out_real, label_real)
            d_x = out_real.mean().item()

            fake = netG(fixed_noise)
            out_fake_1 = netD(fake.detach())
            errD_fake = criterion(out_fake_1, label_fake)
            d_g_z1 = out_fake_1.mean().item()

            errD = (errD_real + errD_fake).item()

            out_fake_2 = netD(fake)
            errG = criterion(out_fake_2, label_real).item()
            d_g_z2 = out_fake_2.mean().item()

        rows.append(LossRow(epoch=epoch, loss_d=float(errD), loss_g=float(errG), d_x=float(d_x), d_g_z1=float(d_g_z1), d_g_z2=float(d_g_z2)))
        print(f"epoch={epoch:02d} loss_d={errD:.4f} loss_g={errG:.4f}")

    if not rows:
        raise RuntimeError("No paired netG/netD checkpoints were evaluated")

    csv_path = os.path.join(out_dir, "loss_by_epoch.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,loss_d,loss_g,d_x,d_g_z1,d_g_z2\n")
        for r in rows:
            f.write(f"{r.epoch},{r.loss_d:.6f},{r.loss_g:.6f},{r.d_x:.6f},{r.d_g_z1:.6f},{r.d_g_z2:.6f}\n")

    epochs_x = [r.epoch for r in rows]
    loss_d_y = [r.loss_d for r in rows]
    loss_g_y = [r.loss_g for r in rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs_x, loss_d_y, label="Loss_D")
    plt.plot(epochs_x, loss_g_y, label="Loss_G")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss (fixed batch)")
    plt.title("DCGAN losses across saved epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plot_path = os.path.join(out_dir, "loss_by_epoch.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
