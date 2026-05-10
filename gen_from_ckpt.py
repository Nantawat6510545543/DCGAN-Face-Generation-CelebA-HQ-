import argparse
import os

import torch
import torchvision.utils as vutils

import main


def _pick_device(device: str) -> torch.device:
    d = device.lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    if d == "auto":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    raise ValueError(f"Unknown --device: {device}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Generate images from a DCGAN netG checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to netG_epoch_*.pth")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for noise")
    parser.add_argument("--n", type=int, default=64, help="Number of images")
    parser.add_argument("--nrow", type=int, default=8, help="Grid columns")
    parser.add_argument("--nz", type=int, default=100, help="Latent size")
    parser.add_argument("--ngf", type=int, default=64, help="Generator feature maps")
    parser.add_argument("--nc", type=int, default=3, help="Channels")

    args = parser.parse_args()

    device = _pick_device(args.device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    netG = main.Generator(nz=args.nz, ngf=args.ngf, nc=args.nc, ngpu=1).to(device)
    state = torch.load(args.ckpt, map_location=device)
    netG.load_state_dict(state)
    netG.eval()

    torch.manual_seed(args.seed)
    fixed_noise = torch.randn(args.n, args.nz, 1, 1, device=device)

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()

    vutils.save_image(fake, args.out, nrow=args.nrow, normalize=True, value_range=(-1, 1))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main_cli()
