# Local acceptance on 9 November 2026

This is software-packaging acceptance evidence for v0.2.0, not an additional
biological validation experiment. Publication transport and CI status are
separate checks from model equivalence.

## Installation and interfaces

- Source installation and a built wheel both install the `rlbppred` command.
- The wheel is installed normally into `site-packages`, not tested only through
  an editable checkout. Prediction is exercised from an unrelated directory.
- The independent package does not import the manuscript `pipeline` tree.
- The wheel contains code and package metadata, not training code, frozen paper
  data, embeddings or large model weights.
- 295 offline software tests cover input contracts, output order/precision, inclusive
  thresholds, malformed assets, fixed model identity, device selection, safe
  deserialization boundaries and refusal to overwrite existing outputs.
- Packing matching per-protein tensors and installing a real model bundle into
  a separate cache were exercised; packed vectors/order and installed weight
  checksum were unchanged.

## Actual frozen model verification

The source was the real strict 300-member checkpoint, SHA-256:

```text
72875c9414f9f3705b9437ce2c2b088257810b737464ab2fb332a0df7406aa41
```

After local conversion, all 300 state dictionaries, all fitted scaler values
and their dtypes were compared exactly. No training or parameter fitting was
performed. The generated safe weight file has SHA-256:

```text
e9f2db20805d6212606851a29e69b8d286ba443b8ae0e2f22872060e963ac3b5
```

The original checkpoint remained unchanged. The bundle is stored separately
under ignored `models/strict-saprot-af2-mlp-v1/`; it is not committed into Git.
The release distributes the safe files, not the historical pickle checkpoint.

## Numerical regression scope

Twelve explicitly synthetic 1,280-dimensional feature vectors were used to
exercise the **actual pretrained weights**, including the original float32
standardization, per-member sigmoid and float64 ensemble average. Synthetic
inputs test implementation equivalence, not biological prediction quality.

- The new API and installed CLI were compared with the prior audited saved-
  ensemble predictor, using the same inputs and batch size: maximum absolute
  score difference **0**.
- Changing batch size from 5 to 256 produced a maximum difference of
  **2.80049e-12**, below the prespecified 1e-6 cross-platform tolerance.
- The installed wheel was also tested with NumPy 2.5.4 on the same fixture;
  maximum difference from the NumPy 1.26.4 fixture output was **0**.
- Input features and protected previous output files were not modified.

The environment was macOS ARM64, CPU, Python 3.12.14, PyTorch 2.10.0,
safetensors 0.8.0, with NumPy 1.26.4 and 2.5.4 tested separately. This record
does not assert that Linux/Windows or CUDA CI has already run.

## Real FASTA / AF2 regression

Two real proteins were recovered read-only from the historical snapshot:
IGHMBP2 (P38935, 993 residues) and LIG4 (P49917, 911 residues). Their complete AF2
structures and raw stored embeddings retained the recorded SHA-256 values.
Using pinned Foldseek release 10-941cd33 on macOS ARM64, AA+3Di strings were
identical to the frozen pLDDT-masked tokens at every position in both proteins.

With the original fixed official SaProt checkpoint, fair-esm 2.0.0, PyTorch
2.10.0, Python 3.12.14 and NumPy 2.5.4 on CPU:

- Maximum feature absolute difference from stored embeddings: **5.66244e-7**;
  mean absolute difference: **4.40863e-8**.
- Score from the regenerated P38935 features: **0.9986980404456457**;
  frozen score: 0.9986980370680492.
- Score from the regenerated P49917 features: **0.9892949765920639**;
  frozen score: 0.9892949543396632.
- Maximum score difference was **2.22524e-8**, below the prespecified 1e-6
  packaging tolerance. Neither model weights nor frozen results were changed.

The official encoder omits contact-prediction regression parameters. This unused
head is explicitly disabled; every embedding-producing parameter is required to
load with strict state matching. No random missing embedding layer is retained.

## Scope boundaries

The installed wheel was also run from outside either checkout. The complete
two-protein offline FASTA command passed, with imports from site-packages. A
live AFDB example for P49917 resolved full-length AF2 model_v6, produced features
and scored 0.9912150198221207. That current-structure score is a functional check,
not the frozen paper score (0.9892949543396632); the structure version/hash differ
and are recorded. No frozen score was replaced by the online result.

The full 16,969 human and 17,002 mouse embedding matrices were not available
locally, so the earlier full-proteome acceptance was **not rerun** by this task.
The original 401-candidate list and all published prediction results were left
unchanged.

The two-protein upstream regression does not establish biological accuracy on
new proteins or a repeated whole-proteome result. Current AFDB structures may
differ from historical AF2 files; the actual accession/version/hash is recorded
and sequence coverage checked, rather than assuming equivalence. No CUDA run,
arbitrary-sequence folding service, PyPI publication or archival DOI is claimed.
No manuscript text or review-comment state was changed by this code split.
