# Unsupervised brain-tumour detection: REFLECT across hospitals and protocols

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

### Healthy data from other hospitals

The same model trained on healthy IXI scans (7520 slices from three London
hospitals, same number of updates) instead of tumour-free BraTS slices, and
tested on the same BraTS subjects:

| 5 correction steps | Healthy data: BraTS | Healthy data: IXI |
|---|---|---|
| REFLECT protocol, mean per-slice max Dice | 74.5 | 73.0 |
| REFLECT protocol, pooled max Dice | 81.7 | 79.6 |
| Strict, pooled Dice | 79.6 | 76.1 |
| Strict, mean Dice over tumour slices | 64.1 | 62.8 |
| Strict, pooled AUPRC | 0.865 | 0.828 |
| Strict, tumour-free slices with false positives | 7.7% | 29.0% |

Tumour segmentation transfers across sites: Dice falls by only 1–3 points.
The cost of the site shift is false alarms, which nearly quadruple on
tumour-free slices. REFLECT's single-slice protocol cannot show this, because
every test slice contains a tumour.

Matching each test subject's brain intensity histogram to the IXI training
data before inference (`eval_reflect.py --harmonise`) does not help: false
alarms rise to 33.5% and pooled Dice falls to 75.0. The site shift is not a
difference in intensity distribution alone; contrast, texture or resolution
(many BraTS T2 scans are thick-slice acquisitions resampled to 1 mm) remain
candidates.

### The real cost of the shift: matched false-alarm rates

Each model's Dice-optimal threshold sits at a different false-alarm rate, so
the table above understates the shift. `scripts/operating_points.py` also fixes
thresholds on the validation subjects so that at most 5% of their tumour-free
slices fire, and applies them to the test subjects (single training seed;
376 tumour-free test slices from 41 subjects):

| Healthy training data | AUPRC | Dice, Dice-optimal (FP) | Dice / mean Dice at ≤5% val FP (test FP) |
|---|---|---|---|
| BraTS (in-domain) | 0.865 | 79.6 (7.7%) | 79.6 / 64.3 (9.8%) |
| IXI | 0.828 | 76.1 (29.0%) | 72.0 / 54.3 (7.4%) |
| IXI, thick-slice augmentation | 0.862 | 77.7 (42.3%) | 64.4 / 45.8 (8.2%) |
| IXI, + 10 epochs on IXI only (control) | 0.828 | 75.8 (31.1%) | 74.7 / 58.6 (17.0%) |
| IXI, + 10 epochs with 5 local subjects | 0.829 | 76.4 (27.7%) | 72.3 / 54.6 (8.8%) |
| IXI, + 10 epochs with 20 local subjects | 0.833 | 76.9 (23.7%) | 76.4 / 60.1 (12.2%) |
| IXI, + 10 epochs with 50 local subjects | 0.834 | 76.8 (22.9%) | 76.2 / 60.2 (9.6%) |

- At a false-alarm rate near the in-domain one, training on IXI costs about
  8 points of pooled Dice and 10 points of mean per-slice Dice, not 1–3.
- Fine-tuning on the tumour-free slices of a few dozen local subjects
  (`scripts/make_mix.py`, oversampled ×10, lr 5e-5, about 30 minutes) recovers
  roughly half of that gap at a similar false-alarm rate.
- Simulated thick-slice acquisitions (`export_reflect_ixi.py --thick 1,...,6`:
  slab averaging along a random axis, resampled to 1 mm) raise AUPRC almost to
  the in-domain value but make the model flag more healthy tissue, so it is
  worst at a matched false-alarm rate.
- Validation false-alarm rates transfer only roughly to the test subjects
  (e.g. 5% on validation gives 7–17% on test).

### Where the false alarms are

`scripts/fp_analysis.py` and `scripts/fp_figure.py`
(`reported/fp_analysis/`). The IXI model's false alarms are bright regions
(mean intensity 0.45 against 0.27 for the brain) and only about 10% lie near
the brain border. They are not explained by through-plane resolution
(Spearman ρ = 0.19, p = 0.24 against a per-subject sharpness proxy), by
distance to the tumour (21.5% of slices fire even in subjects whose 20 central
slices contain no tumour), or by ventricle size. Visually they cover
ventricles enlarged or displaced by mass effect, low frontal slices and
distorted anatomy, which the model trained on tumour-free slices of the BraTS
patients reconstructs without complaint. The in-domain result therefore
benefits from the patients' own anatomy appearing in its "healthy" training
data, and the local slices used for fine-tuning carry the same advantage.

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
