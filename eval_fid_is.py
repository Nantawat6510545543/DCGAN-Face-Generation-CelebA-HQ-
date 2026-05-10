import argparse
import json
import os
from dataclasses import asdict, dataclass

import torch
import torchvision.transforms as transforms

import main


@dataclass
class FidIsResult:
    ckpt: str
    dataroot: str
    num_samples: int
    batch_size: int
    image_size: int
    device: str
    fid: float
    inception_score_mean: float
    inception_score_std: float


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


def _to_01(x: torch.Tensor) -> torch.Tensor:
    # Training tensors are normalized to [-1, 1]. Convert to [0, 1].
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Compute FID and Inception Score for a netG checkpoint")
    parser.add_argument("--dataroot", required=True, help="Path to image folder (flat or ImageFolder-style)")
    parser.add_argument("--ckpt", required=True, help="Path to netG_epoch_*.pth")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument("--num-samples", type=int, default=1000, help="How many real and fake images to use")
    parser.add_argument("--batchSize", type=int, default=64, help="Batch size for real/fake processing")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (Windows: recommend 0)")
    parser.add_argument("--imageSize", type=int, default=64, help="Resize/center-crop size")
    parser.add_argument("--nz", type=int, default=100, help="Latent size")
    parser.add_argument("--ngf", type=int, default=64, help="Generator feature maps")
    parser.add_argument("--nc", type=int, default=3, help="Channels")
    parser.add_argument("--seed", type=int, default=123, help="Seed for noise")

    args = parser.parse_args()

    device = _pick_device(args.device)

    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True, splits=10).to(device)

    # Dataset (same preprocessing as training)
    tfm = transforms.Compose(
        [
            transforms.Resize(args.imageSize),
            transforms.CenterCrop(args.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Use the same logic as training: ImageFolder if subfolders exist, else FlatFolderDataset.
    if main._dir_has_subdirs(args.dataroot):
        import torchvision.datasets as dset

        dataset = dset.ImageFolder(root=args.dataroot, transform=tfm)
    else:
        dataset = main.FlatFolderDataset(root=args.dataroot, transform=tfm)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=False,
    )

    # Load generator
    netG = main.Generator(nz=args.nz, ngf=args.ngf, nc=args.nc, ngpu=1).to(device)
    state = torch.load(args.ckpt, map_location=device)
    netG.load_state_dict(state)
    netG.eval()

    # Update real features
    real_seen = 0
    with torch.no_grad():
        for real_batch, _ in dataloader:
            if real_seen >= args.num_samples:
                break
            real_batch = real_batch.to(device)
            remaining = args.num_samples - real_seen
            if real_batch.size(0) > remaining:
                real_batch = real_batch[:remaining]

            real_imgs_01 = _to_01(real_batch)
            fid_metric.update(real_imgs_01, real=True)
            real_seen += int(real_batch.size(0))

    # Update fake features + inception score
    torch.manual_seed(args.seed)
    fake_seen = 0
    with torch.no_grad():
        while fake_seen < args.num_samples:
            current = min(args.batchSize, args.num_samples - fake_seen)
            noise = torch.randn(current, args.nz, 1, 1, device=device)
            fake = netG(noise)
            fake_imgs_01 = _to_01(fake)
            fid_metric.update(fake_imgs_01, real=False)
            is_metric.update(fake_imgs_01)
            fake_seen += int(current)

    fid_value = float(fid_metric.compute().item())
    is_mean, is_std = is_metric.compute()

    result = FidIsResult(
        ckpt=os.path.abspath(args.ckpt),
        dataroot=os.path.abspath(args.dataroot),
        num_samples=int(args.num_samples),
        batch_size=int(args.batchSize),
        image_size=int(args.imageSize),
        device=str(device),
        fid=fid_value,
        inception_score_mean=float(is_mean.item()),
        inception_score_std=float(is_std.item()),
    )

    out_path = args.out
    if out_path is None:
        ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]
        out_path = os.path.join(os.path.dirname(args.ckpt), f"fid_is_{ckpt_base}.json")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    print("FID:", result.fid)
    print("IS_mean:", result.inception_score_mean)
    print("IS_std:", result.inception_score_std)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main_cli()
