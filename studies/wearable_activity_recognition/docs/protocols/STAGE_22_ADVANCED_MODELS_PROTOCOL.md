# Stage 22: Four Advanced Temporal Architectures

## Purpose and boundaries

Implement four new PyTorch temporal classifiers before scientifically comparing them on HARTH. This stage is a detailed **engineering validation**, not a matched development model benchmark. The four mandatory new model families are Temporal Convolutional Network (TCN), Multi-Scale CNN (InceptionTime-inspired), 1D-ResNet, and Time-Series Transformer.

## Common architecture interface

Each model receives `float32 (batch, 6, 100)` complete IMU windows and produces eight unnormalised class logits. CrossEntropyLoss maps original HARTH labels 1–8 to 0–7 internally; the external dataset labels are never changed. The shared factory, trainable-parameter counter and model definitions live in `code/python/src/core/advanced_temporal_models.py`.

- **TCN:** four residual dilated-convolution blocks (dilations 1/2/4/8), 5-point kernels, channel normalisation, GELU, noncausal full-window classification.
- **Multi-Scale CNN:** InceptionTime-inspired parallel 3/7/15 temporal kernels and max-pooling branch, with 1-point bottleneck and residual connections. Not a claim of reproducing the full published InceptionTime ensemble.
- **1D-ResNet:** an initial temporal stem, three residual convolution blocks, stride-2 middle block, global pooling.
- **Transformer:** 48-dimensional time-step embeddings, sinusoidal positional encoding, two encoder layers, four attention heads, global mean pooling. Bidirectional attention, not a streaming causal system and not inherently interpretable.

## Stage 22 notebook structure

`notebooks/22_advanced_temporal_architectures.ipynb` contains ten substantive sections covering architecture theory, tensor interface, model capacity, small illustrative CPU inference timing, actual source HARTH sample loading, Train-only scaling, one genuine PyTorch optimiser step per new model on actual source windows, engineering limitations, explicit Stage 23 comparison controls and a final Quality Gate.

All Markdown section names use `**bold**` rather than heading hashes. Every significant code cell starts with a numbered banner comment and numbers its internal steps. Plots, printed outputs and Markdown are descriptive and interpretation-bounded. The historical Stage 21 notebook is never overwritten.

**Smoke data:** four official development participants S006/S008/S012/S013, with 32 correctly windowed eligible two-second samples each. Only S006/S008/S012 go to Train and S013 to engineering Validation. **S023/S032 are never loaded.** This tiny sample is not designed to contain all classes or estimate Macro F1; do not describe it as model-performance evidence. Source files may be downloaded from the pinned official public repository if they are absent from the local ignored cache.

**Evaluation scope:** one finite-loss forward/backward/Adam update per model; held-out-person inference shape, finite logits, changed weights; trainable-parameter counts and illustrative CPU forward timing. The larger comprehensive, participant-held-out, matched multi-model scientific evaluation is explicitly reserved for Stage 23.

## Tests and data reliability

`tests/test_advanced_temporal_models.py` verifies four-class factory selection, output shapes, finite logits, gradients, real parameter updates, a shorter valid window and a one-window evaluation batch. The Transformer guards positional maximum length and incompatible attention-head dimensions. These tests are synthetic engineering checks; they do not require HARTH data or demonstrate generalisation.

## Running on user's Windows Cursor machine

```powershell
cd D:\human-movement-intelligence
git pull origin main
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_advanced_temporal_models.py" -v
```

Open `studies/wearable_activity_recognition/notebooks/22_advanced_temporal_architectures.ipynb`, Restart Kernel, Run All Cells, visually review all outputs and the final Quality Gate, save and push executed outputs. Stage 23 begins only after those results have been inspected. Full-benchmark metric claims and Test results do not yet exist.

## Important comparison considerations for Stage 23

Use source-matched source-window IDs, identical participant fold allocation, fixed eight-class metric denominator, Train-only scaling and loss weights, explicit computation budget, per-class F1/support and per-person metrics. Record trained weights and model-architecture configuration, not only scores. Any model that cannot complete under CPU limits should be transparently reported rather than silently omitted or replaced by guessed performance.
