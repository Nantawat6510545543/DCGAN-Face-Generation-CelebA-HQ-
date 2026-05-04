# DCGAN training script (`main.py`)

This folder contains a PyTorch implementation of a **DCGAN** (Generator + Discriminator) for 64x64 RGB image generation.

- Training script: `imagen/main.py`
- Checkpoint/loss evaluation (optional): `imagen/eval_checkpoints.py`
- Generate images from a checkpoint (optional): `imagen/gen_from_ckpt.py`

## What `main.py` does
`main.py` trains a GAN with two networks:
- **Generator (G)**: converts random noise `z` into a fake image.
- **Discriminator (D)**: predicts whether an image is real or fake.

During training it will:
- Load a dataset (`--dataset ...` + `--dataroot ...`)
- Train for `--niter` epochs
- Save sample image grids (real + generated)
- Save model checkpoints (`netG_epoch_*.pth`, `netD_epoch_*.pth`) into `--outf`

## Requirements
A working Python + PyTorch install is required.

Typical install (example only):
- `pip install torch torchvision tqdm`

Tip: if you use a virtual environment, activate it first so `python` and `pip` point to the same environment.

## Quick start (FakeData)
FakeData is useful to verify your environment without downloading datasets.

Run from the `imagen/` folder:

```bash
python .\main.py --dataset fake --outf .\out_test --dry-run --device auto --progress
```

## Train on a real image folder (CelebA-HQ / flat folder)
This repo supports **flat image folders** (no class subfolders) via an internal `FlatFolderDataset` fallback.

Example (CelebA-HQ images inside `CelebAMask-HQ\\CelebA-HQ-img`):

```bash
python .\main.py --dataset folder --dataroot .\CelebAMask-HQ\CelebA-HQ-img --outf .\out_celeba_run2 --niter 25 --batchSize 64 --workers 0 --device auto --progress
```

Notes:
- Images are resized/cropped to **64x64** (`--imageSize 64`).
- If your folder **has subfolders**, `ImageFolder` is used; otherwise the flat-folder loader is used.

## Outputs (in `--outf`)
`main.py` writes artifacts to the output folder:

- `real_samples.png`: a grid of real images from the dataloader
- `fake_samples_epoch_XXX.png`: generated samples (saved periodically)
- `netG_epoch_N.pth`: Generator checkpoint at epoch `N`
- `netD_epoch_N.pth`: Discriminator checkpoint at epoch `N`

## GPU / device selection
Use `--device`:
- `--device cpu` forces CPU
- `--device cuda` forces CUDA (errors if CUDA is not available)
- `--device auto` picks CUDA if available, otherwise CPU

The script prints the selected device (and GPU name when using CUDA).

## Progress bar
Add `--progress` to show a `tqdm` progress bar (requires `tqdm`).

## Windows DataLoader note
On Windows, multiprocessing dataloading can be fragile depending on environment.

- Default workers are set to **0** on Windows.
- If you set `--workers > 0`, ensure you run the script normally (not from inside an interactive child process). If you see dataloader spawn issues, use `--workers 0`.

## Resuming / continuing training
You can continue from checkpoints:
- `--netG path\\to\\netG_epoch_XX.pth`
- `--netD path\\to\\netD_epoch_XX.pth`

Example:

```bash
python .\main.py --dataset folder --dataroot .\CelebAMask-HQ\CelebA-HQ-img --outf .\out_celeba_run2 --niter 5 --netG .\out_celeba_run2\netG_epoch_24.pth --netD .\out_celeba_run2\netD_epoch_24.pth --device auto --progress
```

## Generate images from a saved Generator checkpoint
Use `gen_from_ckpt.py`:

```bash
python .\gen_from_ckpt.py --ckpt .\out_celeba_run2\netG_epoch_24.pth --out .\generated_final_epoch.png --device auto --seed 123
```

## GitHub note (recommended)
The dataset archives, extracted images, and training outputs can be very large.
If you publish this repo, consider adding these to `.gitignore`:
- `imagen/CelebAMask-HQ/`
- `imagen/out*/`
- `imagen/*.zip`
- `training_checkpoints/`

## Reference
This implementation is based on the standard DCGAN architecture popularized by the PyTorch DCGAN tutorial and example code.
