# Strict SaProt-AF2 MLP ensemble

Maintainer and publisher: callmest. Model ID: `strict-saprot-af2-mlp-v1`.

Software [v0.2.1](https://github.com/callmest/RLBPPred/releases/tag/v0.2.1)
uses the unchanged pretrained files from
[model release v0.2.0](https://github.com/callmest/RLBPPred/releases/tag/v0.2.0).
Training and paper reproduction are in
[callmest/RLBPPred-paper-code](https://github.com/callmest/RLBPPred-paper-code).

## Intended use

Research prioritization of possible R-loop-associated/binding proteins from
exactly matching SaProt-650M-AF2 global-mean representations. Not a clinical
test or proof of direct physical binding. No training occurs during installation,
feature extraction or prediction.

## Training provenance and fixed inference

Curated benchmark: 376 positive and 9,109 negative proteins. The fixed stratified
split uses 301 positive/7,288 negative training proteins and 75 positive/1,821
negative held-out proteins (seed 42). A training-only scaler is fitted to the
7,589 training proteins. Each of 300 balanced rounds uses all 301 training
positives and 301 sampled training negatives. This release reuses the already
trained weights; it is not a newly trained or selected variant.

Each member has 1,280→512→256→1 linear layers, batch normalization, ReLU and
dropout 0.3. In evaluation mode, the release preserves float32 standardization,
per-member sigmoid and float64 ensemble accumulation. Scores are not calibrated
posterior probabilities of biological function.

The 0.98 threshold yields an empirical FPR of 1/1,821 (0.0549%) on the particular
held-out human negatives. It meets the empirical <=0.1% target on that split,
not an upper confidence bound or guaranteed deployment FPR/FDR. The 401 paper
candidates refer to the separate original 5,640-protein discovery pool. Do not
interpret all high-scoring known training proteins as novel candidates.

## Assets

The Release supplies `model.json` (1,012 bytes) and `model.safetensors`
(949,122,160 bytes; approximately 905 MiB). The latter contains all 300 members
and the scaler, not the SaProt encoder. File hashes are pinned in the downloader
and published independently in the release checksum file. Original checkpoint
SHA-256: `72875c9414f9f3705b9437ce2c2b088257810b737464ab2fb332a0df7406aa41`.
Safe weights SHA-256: `e9f2db20805d6212606851a29e69b8d286ba443b8ae0e2f22872060e963ac3b5`.
MLP weights are MIT licensed, without warranty.

## Limits

Domain shifts, annotation biases, incomplete labels, species transfer and
dependence on structure quality can affect interpretation. AFDB availability
is not guaranteed; arbitrary FASTA without a matching AF2 structure cannot be
processed by this wrapper. Experimental structures and AF3 models are not
silently treated as AF2. No sequence-only fallback, fragment merging, implicit
truncation or long-sequence chunk averaging is used. CPU biological regression
and software tests are described in VALIDATION.md; no CUDA equivalence claim or
full human/mouse rerun is made by this packaging release.
