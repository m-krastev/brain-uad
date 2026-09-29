# Unsupervised brain-tumour detection: REFLECT under a stricter protocol

Unsupervised anomaly detection learns what healthy brains look like and flags
what a model cannot map back to healthy anatomy. REFLECT (Beizaee et al.,
MICCAI 2025) does this with a rectified flow in the latent space of a medical
VAE, correcting an abnormal scan towards a healthy one in a single step. This
repository reproduces REFLECT on BraTS 2021 T2 and asks how much of its result
depends on the evaluation protocol and on where the healthy training data
comes from.

## Results (BraTS 2021, T2)

REFLECT-1, UNet-M, KL-f8 VAE, trained with the paper's settings (200 epochs,
batch 96, lr 5e-4) on tumour-free central slices of the BraTS training
subjects; 5 correction steps unless noted.

| Protocol | Dice | Notes |
|---|---|---|
| Paper, REFLECT protocol | 79.6 | |
| Ours, REFLECT protocol, pooled max Dice | 81.7 | 1 step: 80.8 |
| Ours, REFLECT protocol, mean per-slice max Dice | 74.5 | 1 step: 73.7 |
| Ours, strict, pooled Dice | 79.6 | threshold from validation subjects |
| Ours, strict, mean Dice over tumour slices | 64.1 | |

REFLECT protocol: one slice per test subject (the one with the largest
tumour, 116 subjects), threshold swept on the test slices. Strict protocol:
all 20 central slices of each test subject (2520 slices, 2144 with tumour),
threshold fixed on the validation subjects' slices, anomaly maps restricted to
the brain.

- The result reproduces: pooled Dice slightly above the paper, per-slice Dice
  a few points below; the paper does not state which aggregate it reports.
- Choosing the threshold on the test set costs almost nothing here: the
  validation threshold gives 79.6 against 79.7 with the test-optimal one.
- Scoring only the most pathological slice hides the hard cases. Pooled Dice
  barely changes, because large tumours dominate the pixel count, but the mean
  Dice per tumour slice falls by about 10 points once every slice counts, and
  7.7% of tumour-free slices contain false-positive pixels (13.0% with one
  step), which the single-slice protocol never measures.

IXI_PLACEHOLDER

## Setup

- **BraTS 2021** (1251 scans; T2 and whole-tumour masks), split 80/10/10 by
  subject (seed 0) as in REFLECT. 20 central axial slices per scan, brain
  1st–99th intensity percentiles mapped to 0–255, 240 × 240 in SRI24/BraTS
  space, padded to 256 (`scripts/export_reflect.py`). REFLECT registers to
  MNI152 instead; its exact scaling is not published.
- **IXI** (578 healthy T2 scans from three London hospitals): HD-BET skull
  stripping (`scripts/run_hdbet.sh`), affine registration to SRI24 with ANTs,
  same slicing and scaling (`scripts/export_reflect_ixi.py`).
- **REFLECT** is run from the authors' code
  ([farzad-bz/REFLECT](https://github.com/farzad-bz/REFLECT), commit 6e05909)
  with `patches/reflect_0001_16gb_and_torch26.patch`: full loading of the
  pickled VAE under PyTorch ≥ 2.6, CPU map-location for evaluation, VAE
  encoding in chunks of 16 and U-Net updates in micro-batches of 48 with
  size-weighted losses (identical gradients, GroupNorm only) to fit 16 GB, and
  a fix to the released test loader, which returns an undefined variable.
- `scripts/eval_reflect.py` scores a checkpoint under both protocols with
  REFLECT's own anomaly map.

## Earlier baselines (not reproduced)

`uad/` and `scripts/train.py` / `evaluate.py` contain my reimplementations of
an autoencoder, AnoDDPM-style DDPM and patched DDPM (pDDPM, Behrendt et al.,
MIDL 2023) on 96 × 96 volumes with pDDPM's preprocessing
(`scripts/preprocess.py`). They do not reproduce the published numbers: the
DDPM reaches 22.6 Dice on BraTS 2021 against 40.7 reported, after matching
the reference code's noise scale, intensity scaling and U-Net widths. They are
kept for transparency and are not used in any comparison.

## Running

```bash
uv sync                                        # this repo's env
bash scripts/run_hdbet.sh                      # IXI skull stripping
python scripts/export_reflect.py               # BraTS 2021 slices
python scripts/export_reflect_ixi.py           # IXI slices
# REFLECT: clone into third_party/REFLECT, apply the patch, build its env
torchrun train_REFLECT.py --dataset BraTS --model UNet_M --vae kl_f8 --modality T2 \
    --lr 5e-4 --global-batch-size 96 --epochs 200 --data-dir ../../data/reflect_brats21_t2
third_party/REFLECT/.venv/bin/python scripts/eval_reflect.py --ckpt CKPT \
    --data data/reflect_brats21_t2 --steps 5 --out results/reflect_brats21
```

## Licence and data

Apache-2.0 for this repository's code. REFLECT's code is not redistributed
(no licence file); only the patch is included. IXI (CC BY-SA 3.0) and
BraTS 2021 (cite the challenge papers) are not redistributed.
