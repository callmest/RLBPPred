# RLBPPred: pretrained inference

Predict R-loop-binding protein scores with the frozen **strict SaProt-650M-AF2 +
300-member MLP ensemble**. This repository is an independently installable
prediction tool; it does not train a new model. Training, paper figures, frozen
results and downstream analyses are in [callmest/RLBPPred](https://github.com/callmest/RLBPPred).

## FASTA to predictions: three commands

Use Python 3.10–3.13 in a fresh environment on macOS or Linux x86-64 (AVX2).
Install from this versioned GitHub repository; no PyPI publication is implied.

```bash
python -m pip install "rlbppred[fasta] @ git+https://github.com/callmest/RLBPPred-inference.git@v0.2.0"
rlbppred setup
rlbppred predict-fasta --fasta proteins.fasta --output-dir predictions
```

`setup` explicitly downloads approximately 3.6 GB once: our safe pretrained MLP
bundle from the [v0.2.0 GitHub Release](https://github.com/callmest/RLBPPred-inference/releases/tag/v0.2.0),
the official SaProt checkpoint from a fixed Hugging Face revision, and a pinned
Foldseek executable. Sizes and SHA-256 hashes are checked before use. Large
weights are not stored in ordinary Git or the code wheel. No Zenodo deposit is
required. Subsequent runs reuse valid local assets; corrupted files are rejected,
not silently replaced. See [asset versions and manual downloads](docs/assets.md).

The FASTA command performs:

1. Retrieve an exact full-length, sequence-matched AF2 structure from AlphaFold DB
   for each UniProt accession (e.g. `P49917`, or `sp|P49917|...`).
2. Convert the selected chain to AA+3Di with Foldseek; mask 3Di as `#` where the
   mean per-residue atomic pLDDT is below 70.
3. Extract raw float32 1,280-dimensional features: final SaProt layer, global
   residue mean excluding BOS/EOS. Use the original fair-esm backend, not ESM-C,
   SaProt-PDB, a sequence-only fallback or another pooling.
4. Apply the fixed training scaler and all 300 pretrained MLP members, then save
   the mean of per-member sigmoid scores.

No protein sequence is submitted to a folding service. AFDB requests use only
the accession. An arbitrary sequence with no exact AFDB entry needs a matching
local **AF2** structure; structure prediction is not secretly launched for it.
The software never substitutes another sequence, silently truncates a protein,
merges structure fragments, chooses an ambiguous chain or skips a failed input.
It stops with an explanation and preserves the partial run for diagnosis.
CPU is the default; long proteins can require substantial memory and time.
For a compatible PyTorch/CUDA installation, pass `--device cuda:0`. An unavailable
device fails instead of silently switching backends. CUDA was not exercised in
the local packaging acceptance.

### Local structures and offline prediction

Place `ID.pdb` or `ID.cif` in a directory; the complete selected chain must match
the FASTA exactly. An AFDB export carries an AlphaFold header. A custom AF2 export
without a source header additionally needs `ID.pdb.json` (or `ID.cif.json`):

```json
{"structure_source": "alphafold2", "bfactor_is_plddt": true}
```

Only make that declaration when the structure really is AF2 and its B-factors
really hold pLDDT. Experimental temperature factors are not pLDDT.

```bash
rlbppred predict-fasta --fasta proteins.fasta --structure-dir af2_structures \
  --offline --output-dir offline_predictions
```

Offline prediction needs previously downloaded assets. `--assets-dir`,
`--model-dir`, `--saprot-model` and `--foldseek` override their locations;
`RLBPPRED_ASSETS_DIR` and `RLBPPRED_MODEL_DIR` change cache defaults. The pLDDT
cutoff is fixed at 70 in this workflow. An optional `--max-residues` rejects inputs
above an explicit resource limit; it does not truncate them.

### Outputs

A new output directory contains `predictions.csv`, `features.npz`, original
structure metadata, masked `tokens/tokens.fasta`, checksums and `run.json`.
Existing output directories or predictions are never overwritten.
The CSV preserves input order and full-precision scores:

| Column | Meaning |
|---|---|
| `protein_id` | Input identifier; UniProt FASTA headers normalize to accession |
| `probability` | Frozen ensemble score, not a calibrated biological truth probability |
| `classification_at_0_5` | Score >= 0.5 |
| `above_screening_threshold` | Score >= 0.98 by default |

`--threshold` changes only the screening flag, not the model or scores. The 401
paper candidates came from a specific 5,640-protein untrained human pool.
Arbitrary high-scoring inputs are **not automatically novel candidates**. The
human held-out empirical FPR of 1/1,821 (0.0549%) is not a guaranteed deployment
FPR/FDR, and mouse deployment FPR was not established by that experiment.

## Score existing matching embeddings

For MLP-only installation omit the `fasta` extra:

```bash
python -m pip install "rlbppred @ git+https://github.com/callmest/RLBPPred-inference.git@v0.2.0"
rlbppred setup --components model
rlbppred predict --embeddings features.npz --output predictions.csv
```

The safe NPZ must have a nonempty unique 1-D string `protein_id` array and finite
**raw, unstandardized float32 `N x 1280`** `X`. The bundle applies its own fixed
scaler. Do not standardize again. Numeric validation cannot prove the upstream
provenance of an arbitrary matrix. Object/pickle NPZ arrays are rejected.

Legacy matching per-protein tensor dictionaries can be packed without retraining:

```bash
rlbppred pack --embedding-dir embeddings --ids protein_ids.txt --output features.npz
rlbppred model verify
rlbppred doctor
```

Packing requires `model=saprot-650m-af2` and `pooling=mean` metadata and uses the
restricted tensor loader. The code-only wheel has no neighboring-repository
dependency, training code or model weights.

## Model identity and reproducibility

The released MLP is converted from the existing strict checkpoint (SHA-256
`72875c9414f9f3705b9437ce2c2b088257810b737464ab2fb332a0df7406aa41`).
All 300 member states and fitted scaler values are preserved. Normal inference
uses safetensors, not pickle. The fixed SaProt download uses revision
`1b8cddd7bad34f6f5627422b4908309840c78fef`; its SHA-256 is
`bc110567bc75511c7f46eb3e7d109fd6e81051baad471e22c5306ed71653cd52`.

See [validation evidence](VALIDATION.md), [model card](MODEL_CARD.md) and
[asset locks](docs/assets.md). Real AF2-structure regression on IGHMBP2 (P38935)
and LIG4 (P49917) produced exactly the frozen AA+3Di tokens; feature differences
were below 1e-6 and score differences below 1e-7. These checks establish packaging
equivalence on two inputs, not additional biological validation or a repeated
whole-proteome benchmark. Current AFDB structures may differ from the frozen
paper snapshot; each run records its actual version and hash.

## Development

```bash
git clone https://github.com/callmest/RLBPPred-inference.git
cd RLBPPred-inference
python -m pip install '.[dev,fasta]'
python -m pytest
python -m build
```

Default tests are offline software fixtures, with no large downloads. They cover
invalid inputs, structure matching, asset integrity, safe loading, all-member
aggregation, failure receipts and refusal to overwrite results. They do not run
a new training job. Original code and MLP weights are MIT licensed; third-party
software, encoder weights and structure data retain their own terms. See
[third-party notices](THIRD_PARTY_NOTICES.md). Maintainer: **callmest**.
