# Fixed assets and offline setup

`rlbppred setup` is the only automatic weight-download step. Prediction downloads
only explicitly permitted AFDB structures when local structures are absent.
Weights are never trained or replaced automatically. The setup command checks
the following fixed sizes and SHA-256 values, including existing cached files.

| Asset | Bytes | SHA-256 |
|---|---:|---|
| SaProt_650M_AF2.pt | 2,606,464,143 | `bc110567bc75511c7f46eb3e7d109fd6e81051baad471e22c5306ed71653cd52` |
| model.safetensors | 949,122,160 | `e9f2db20805d6212606851a29e69b8d286ba443b8ae0e2f22872060e963ac3b5` |
| model.json | 1,012 | `82458d98d41b694861c24911a8554564e59a49adf471eb034a44d8da94d9c247` |
| Foldseek macOS archive | 17,021,264 | `6d8b07e188d443044f0c98db30aa3d4cca6750e97b50679372b54b45a664cae5` |
| Foldseek Linux AVX2 archive | 11,941,946 | `e6a3e3860e76b0b7b858736f627fe680d4d7f0fc7298301d8a92b9f711d74ac7` |

## Official encoder

The exact upstream version is `westlake-repl/SaProt_650M_AF2`, revision
`1b8cddd7bad34f6f5627422b4908309840c78fef`. Only the original fair-esm `.pt`
checkpoint is needed; do not also download the alternative transformers weights.

```bash
curl --fail --location \
  'https://huggingface.co/westlake-repl/SaProt_650M_AF2/resolve/1b8cddd7bad34f6f5627422b4908309840c78fef/SaProt_650M_AF2.pt' \
  --output SaProt_650M_AF2.pt
```

Verify SHA-256 (e.g. `sha256sum` on Linux or `shasum -a 256` on macOS), then pass
`--saprot-model SaProt_650M_AF2.pt`. The prediction loader independently checks
its pinned size/hash before restricted `weights_only=True` loading. No arbitrary
pickle fallback, alternative encoder, incomplete parameter loading or random
initialization of embedding layers is permitted. Only the unused contact
prediction head, absent from this official checkpoint, is explicitly disabled.

## Strict MLP

Download these two files from the
[v0.2.0 Release](https://github.com/callmest/RLBPPred-inference/releases/tag/v0.2.0),
not from an untrusted third party:

```bash
mkdir strict-saprot-af2-mlp-v1
curl --fail --location \
  https://github.com/callmest/RLBPPred-inference/releases/download/v0.2.0/model.json \
  --output strict-saprot-af2-mlp-v1/model.json
curl --fail --location \
  https://github.com/callmest/RLBPPred-inference/releases/download/v0.2.0/model.safetensors \
  --output strict-saprot-af2-mlp-v1/model.safetensors
rlbppred model install --source strict-saprot-af2-mlp-v1
```

Do not commit these large files into Git. A local code wheel does not embed
them. `rlbppred setup --components model` is the simpler equivalent with fixed
download hashes. MLP weights are below GitHub's per-Release-asset limit but above
the ordinary-Git file limit. No Zenodo host or DOI is part of this release.

## Foldseek

The setup command uses release **10-941cd33**, executable version
`941cd33ff0771cd2e3f144e3293e22a2b87e9fda`:

- [macOS universal archive](https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-osx-universal.tar.gz).
  Binary SHA-256 `1b446bbece6b01cf0a50b28419f353522204f44a10a5872462f8ad400d9aadab`.
- [Linux x86-64 AVX2 archive](https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-avx2.tar.gz).
  Binary SHA-256 `4db94a7825d409e416643772860ddf7cfe328d582679b47e6dffda7df4d41ea0`.

Only the checked regular executable is extracted; archive links and unsafe paths
are rejected. A search database and the Foldseek GPU build are not needed for
`structureto3didescriptor`. Other CPU architectures/platforms need a compatible
external executable supplied via `--foldseek`; they are not validated by this
release. An override records the actual version/hash and is not silently claimed
equivalent to the pinned configuration. Linux AVX2 CPU support is a prerequisite.

## Cache and offline use

Defaults are `~/.cache/rlbppred/assets` for encoder/Foldseek and
`~/.cache/rlbppred/strict-saprot-af2-mlp-v1` for the MLP. Use `--assets-dir`,
`--model-dir`, `RLBPPRED_ASSETS_DIR` or `RLBPPRED_MODEL_DIR` to move them.
Downloaded files are streamed and verified before atomic publication. An
existing corrupt file or symbolic link fails; move it aside deliberately before
retrying. Nothing is deleted or overwritten as an automatic repair.

To move to an offline machine, copy the verified assets plus exact AF2 structures
and their metadata. Run `predict-fasta --structure-dir ... --offline ...` there.
Record checksums/versions with results. Current AFDB metadata is saved per run;
the online service is not a frozen structural snapshot.
