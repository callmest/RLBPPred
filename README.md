# RLBPPred

`RLBPPred` is a command-line tool for ranking proteins as candidate R-loop-binding proteins. It scores each protein with a pretrained SaProt-650M-AF2 encoder and a 300-member MLP ensemble. Start with a UniProt FASTA file, a local protein structure, or precomputed SaProt embeddings.

## Installation
`RLBPPred` only support local installation, but will suport PyPI or Bioconda online installation in the future.

Use Python 3.10–3.13. Install [v0.3.0](https://github.com/callmest/RLBPPred/releases/tag/v0.3.0) from source to run the bundled examples:

```
conda create -n rlbppred python=3.10
git clone --branch v0.3.0 --depth 1 https://github.com/callmest/RLBPPred.git
cd RLBPPred
pip install '.[fasta]'
rlbppred setup
```
The `fasta` extra supports both sequence and structure inputs. Setup downloads and verifies about 3.6 GB once. The bundled Foldseek executable supports modern macOS and Linux x86_64. Prediction runs on **CPU** by default.

## Usage
### Examples
1. If your target protein got structure in [AlphaFoldDB](https://alphafold.ebi.ac.uk/)
```
rlbppred predict --fasta examples/P49917.fasta --output-dir results_lig4
```
2. User-defined structure
```bash
rlbppred predict --pdb examples/1UBQ.pdb --chain A --output-dir results_1ubq
rlbppred predict --pdb examples/1UBQ.mmcif --chain A --output-dir results_1ubq
```

### Full usage
```
usage: rlbppred predict [-h] (--fasta FASTA | --pdb PDB | --embeddings EMBEDDINGS) (--output-dir OUTPUT_DIR | --output OUTPUT) [--chain CHAIN]
                        [--structure-source {auto,af2,experimental}] [--assets-dir ASSETS_DIR] [--saprot-model SAPROT_MODEL] [--foldseek FOLDSEEK]
                        [--max-residues MAX_RESIDUES] [--model-dir MODEL_DIR] [--device DEVICE] [--batch-size BATCH_SIZE] [--threshold THRESHOLD]

options:
  -h, --help            show this help message and exit
  --fasta FASTA         UniProt FASTA; retrieve exact matching AF2 structures from AlphaFold DB
  --pdb PDB             Local AF2 or experimental PDB/mmCIF; score coordinates for one chain
  --embeddings EMBEDDINGS
                        NPZ with protein_id and raw float32 X (N x 1280)
  --output-dir OUTPUT_DIR
                        New run directory for --fasta or --pdb
  --output OUTPUT       New predictions CSV for --embeddings
  --chain CHAIN         Protein chain ID for --pdb (required if multiple protein chains)
  --structure-source {auto,af2,experimental}
                        For --pdb only; default: auto from metadata
  --assets-dir ASSETS_DIR
                        Override the default asset cache
  --saprot-model SAPROT_MODEL
                        Override the fixed encoder checkpoint location
  --foldseek FOLDSEEK   Override the Foldseek executable location
  --max-residues MAX_RESIDUES
                        Reject inputs beyond the limit rather than truncate
  --model-dir MODEL_DIR
  --device DEVICE
  --batch-size BATCH_SIZE
  --threshold THRESHOLD
                        Screening flag; not a guaranteed FPR/FDR
```

## Expected results
Open `results_lig4/predictions.csv`. With AlphaFold DB structure version 6, the example produced this row:

| protein_id | probability | classification_at_0_5 | above_screening_threshold |
| :---: | :---: | :---: | :---: |
| P49917 | 0.991215 | 1 | 1 |

The score is rounded here. `probability` is the mean ensemble score, not a calibrated probability of binding. The last two columns flag scores **≥0.5** and **≥0.98** (the default screening cutoff), respectively. A high score prioritizes a protein for follow-up; it does not establish binding.

## Documentations

- [Model card](MODEL_CARD.md) and [validation results](VALIDATION.md) — training data, screening cutoff, and checks against reference predictions.
- [Installation and offline setup](docs/assets.md) — model downloads, versions, checksums, and cache locations.
- [Example inputs](examples/README.md) — file sources and requirements for local structures.
- [RLBPPred-paper-code](https://github.com/callmest/RLBPPred-paper-code) — training and analysis code for reproducing the paper.


## Acknowledgement
`RLBPPred` is dependent on the following libraries, we are grateful to all the developers/maintainers:
- [SaProt](https://github.com/westlake-repl/Saprot): Structure-aware protein language model
- [Pytorch](https://github.com/pytorch/pytorch): Model training
- [BioPython](https://github.com/biopython/biopython): IO


