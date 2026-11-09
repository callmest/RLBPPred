"""Inference-only runtime for the frozen strict SaProt-AF2/MLP ensemble.

The numerical contract follows the verified ``strict_runtime.py`` in the
manuscript reproducibility project, without importing that project or exposing
its training workflows.  The legacy pickle is accepted *only* by ``export_model``
after matching its fixed release SHA-256.  Normal inference loads safetensors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


CHECKPOINT_SHA256 = "72875c9414f9f3705b9437ce2c2b088257810b737464ab2fb332a0df7406aa41"
MODEL_ID = "strict-saprot-af2-mlp-v1"

_ARCHITECTURE = {
    "input_dim": 1280,
    "hidden_dims": [512, 256],
    "dropout": 0.3,
    "batch_norm": True,
    "members": 300,
}
_PREPROCESSING = {
    "scaler_fit": "strict-training-only",
    "scaler_n_samples_seen": 7589,
    "pooling": "global-mean-excluding-special-tokens",
}
_SCALERS = ("scaler_mean", "scaler_scale", "scaler_var")
_STATE_SHAPES = {
    "net.0.weight": (512, 1280),
    "net.0.bias": (512,),
    "net.4.weight": (256, 512),
    "net.4.bias": (256,),
    "net.8.weight": (1, 256),
    "net.8.bias": (1,),
}
for _layer, _width in ((1, 512), (5, 256)):
    for _field in ("weight", "bias", "running_mean", "running_var"):
        _STATE_SHAPES[f"net.{_layer}.{_field}"] = (_width,)
    _STATE_SHAPES[f"net.{_layer}.num_batches_tracked"] = ()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    """Return a file's SHA-256 without loading it entirely into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model export and prediction.") from exc
    return torch


def _safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("Install safetensors to export or load the prediction model.") from exc
    return safe_open, load_file, save_file


def _validate_state(state: Mapping[str, Any], member: int, torch: Any) -> None:
    _require(isinstance(state, Mapping), f"Member {member}: state must be a mapping")
    _require(set(state) == set(_STATE_SHAPES), f"Member {member}: incorrect state keys")
    for name, shape in _STATE_SHAPES.items():
        tensor = state[name]
        label = f"Member {member}: {name}"
        _require(isinstance(tensor, torch.Tensor), f"{label} must be a tensor")
        _require(tuple(tensor.shape) == shape, f"{label}: expected shape {shape}")
        _require(tensor.device.type == "cpu", f"{label}: expected CPU storage")
        if name.endswith("num_batches_tracked"):
            _require(tensor.dtype == torch.int64 and int(tensor) == 500,
                     f"{label}: expected int64 counter 500")
        else:
            _require(tensor.dtype == torch.float32, f"{label}: expected float32")
            _require(bool(torch.isfinite(tensor).all()), f"{label}: non-finite values")
            if name.endswith("running_var"):
                _require(bool((tensor >= 0).all()), f"{label}: negative running variance")


def _validate_scalers(scalers: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    result = {}
    for name in _SCALERS:
        value = scalers[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        _require(array.shape == (1280,), f"{name}: expected shape (1280,)")
        _require(array.dtype in (np.dtype("float32"), np.dtype("float64")),
                 f"{name}: expected float32 or float64")
        _require(bool(np.isfinite(array).all()), f"{name}: non-finite values")
        if name == "scaler_scale":
            _require(bool((array > 0).all()), "Scaler scales must be positive")
        if name == "scaler_var":
            _require(bool((array >= 0).all()), "Scaler variances must be nonnegative")
        # Do not cast: StandardScaler's float64 statistics are significant to
        # the original in-place float32 subtraction/division rounding.
        result[name] = np.array(array, copy=True)
    return result


def _validate_legacy_checkpoint(checkpoint: Any, torch: Any) -> tuple[list[Any], dict[str, Any]]:
    _require(isinstance(checkpoint, Mapping), "Checkpoint must be a mapping")
    expected = {
        "format": "state_dict_ensemble_v1",
        "upstream": "saprot-650m-af2",
        "clf_name": "MLP",
        "in_dim": 1280,
        "hidden_dims": [512, 256],
        "dropout": 0.3,
        "use_bn": True,
        "n_rounds": 300,
        "seed": 42,
        "scaler_n_samples_seen": 7589,
    }
    for name, value in expected.items():
        _require(checkpoint.get(name) == value, f"Checkpoint mismatch: {name}")
    states = checkpoint.get("model_state_dicts")
    _require(isinstance(states, (list, tuple)) and len(states) == 300,
             "Checkpoint must contain exactly 300 member states")
    for member, state in enumerate(states):
        _validate_state(state, member, torch)
    _require(all(name in checkpoint for name in _SCALERS), "Checkpoint is missing scaler arrays")
    return list(states), _validate_scalers(checkpoint)


def export_model(checkpoint: str | Path, output_dir: str | Path) -> Path:
    """Convert the fixed trusted historical checkpoint into an inference bundle.

    This operation does not fit a model, change parameters, or require the
    separate joblib scaler.  Unknown checkpoint hashes are refused *before*
    pickle deserialization.  The destination must not already exist.
    """
    import numpy as np

    source = Path(checkpoint).expanduser().resolve(strict=True)
    requested_output = Path(output_dir).expanduser()
    _require(not requested_output.is_symlink(), f"Output is a symbolic link: {requested_output}")
    output = requested_output.resolve()
    _require(source.is_file(), "Checkpoint is not a regular file")
    _require(not output.exists(), f"Output already exists: {output}")
    _require(sha256(source) == CHECKPOINT_SHA256,
             "Checkpoint SHA-256 mismatch; refusing legacy pickle deserialization")
    torch = _torch()
    _, _, save_file = _safetensors()
    # This exact byte sequence was independently frozen and audited. Do not add
    # an arbitrary --trust/--hash override or expose this loader to other files.
    legacy = torch.load(source, map_location="cpu", weights_only=False)
    states, scalers = _validate_legacy_checkpoint(legacy, torch)
    _require(sha256(source) == CHECKPOINT_SHA256, "Checkpoint changed during export")
    tensors = {}
    for member, state in enumerate(states):
        for name, value in state.items():
            tensors[f"member_{member:03d}.{name}"] = value.detach().cpu().contiguous()
    for name, value in scalers.items():
        tensors[name] = torch.from_numpy(np.ascontiguousarray(value))
    # Exclusive directory creation prevents accidental replacement of an
    # installed bundle. Failed exports remain visibly incomplete (no valid
    # manifest), rather than deleting potentially concurrent user files.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    weights = output / "model.safetensors"
    save_file(tensors, str(weights), metadata={
        "model_id": MODEL_ID,
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
    })
    metadata = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "format": "safetensors_ensemble_v1",
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream": "saprot-650m-af2",
        "classifier": "MLP",
        "seed": 42,
        "architecture": dict(_ARCHITECTURE),
        "preprocessing": dict(_PREPROCESSING),
        "scaler_dtypes": {name: str(value.dtype) for name, value in scalers.items()},
        "weights": {"filename": weights.name, "sha256": sha256(weights)},
        "classification_threshold": 0.5,
        "screening_threshold": 0.98,
        "screening_scope": "original human untrained-proteome pool; not a guarantee for arbitrary inputs",
    }
    with (output / "model.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return output


def _validate_manifest(metadata: Any) -> tuple[str, str]:
    _require(isinstance(metadata, dict), "Model manifest must be a JSON object")
    expected = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "format": "safetensors_ensemble_v1",
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream": "saprot-650m-af2",
        "classifier": "MLP",
        "seed": 42,
        "architecture": _ARCHITECTURE,
        "preprocessing": _PREPROCESSING,
        "classification_threshold": 0.5,
        "screening_threshold": 0.98,
    }
    for name, value in expected.items():
        _require(metadata.get(name) == value, f"Model manifest mismatch: {name}")
    weights = metadata.get("weights")
    _require(isinstance(weights, dict), "Model manifest is missing weights")
    filename, digest = weights.get("filename"), weights.get("sha256")
    _require(isinstance(filename, str) and filename == "model.safetensors",
             "Weights must be the flat filename model.safetensors")
    _require(isinstance(digest, str) and len(digest) == 64
             and all(c in "0123456789abcdef" for c in digest), "Invalid weights SHA-256")
    dtypes = metadata.get("scaler_dtypes")
    _require(isinstance(dtypes, dict) and set(dtypes) == set(_SCALERS)
             and all(value in ("float32", "float64") for value in dtypes.values()),
             "Invalid scaler dtype declaration")
    return filename, digest


def _device(torch: Any, requested: str) -> str:
    _require(isinstance(requested, str), "Device must be a string")
    if requested == "cpu":
        return requested
    if requested == "cuda" or (requested.startswith("cuda:") and requested[5:].isdigit()):
        _require(torch.cuda.is_available(), f"Requested device {requested} is unavailable")
        index = 0 if requested == "cuda" else int(requested[5:])
        _require(index < torch.cuda.device_count(), f"Requested CUDA index {index} is unavailable")
        return requested
    raise ValueError("Device must be cpu, cuda, or cuda:<nonnegative index>; no implicit fallback")


def _build_network(torch: Any, device: str):
    class StrictMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            previous = 1280
            for width in (512, 256):
                layers.extend((torch.nn.Linear(previous, width), torch.nn.BatchNorm1d(width),
                               torch.nn.ReLU(), torch.nn.Dropout(0.3)))
                previous = width
            layers.append(torch.nn.Linear(previous, 1))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    # The API can be imported into a process whose global default dtype differs
    # from the training run; keep this release's layers explicitly float32.
    return StrictMLP().to(device=device, dtype=torch.float32).eval()


@dataclass
class Ensemble:
    """Loaded inference bundle; ``metadata`` records the frozen model contract."""

    metadata: dict[str, Any]
    device: str
    _torch: Any = field(repr=False)
    _network: Any = field(repr=False)
    _states: list[dict[str, Any]] = field(repr=False)
    _scalers: dict[str, Any] = field(repr=False)


def load_model(model_dir: str | Path, device: str = "cpu") -> Ensemble:
    """Verify and load a safe, fixed-architecture inference bundle on CPU/CUDA."""
    directory = Path(model_dir).expanduser().resolve(strict=True)
    _require(directory.is_dir(), "Model bundle must be a directory")
    manifest = directory / "model.json"
    _require(manifest.is_file() and not manifest.is_symlink(), "Model manifest must be a regular non-symlink file")
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Invalid model manifest JSON") from exc
    filename, expected_sha = _validate_manifest(metadata)
    weights = directory / filename
    _require(weights.is_file() and not weights.is_symlink(), "Model weights must be a regular non-symlink file")
    _require(weights.resolve().parent == directory, "Model weights escape bundle directory")
    _require(sha256(weights) == expected_sha, "Model weights SHA-256 mismatch")
    torch = _torch()
    selected_device = _device(torch, device)
    safe_open, load_file, _ = _safetensors()
    from safetensors import SafetensorError
    try:
        with safe_open(str(weights), framework="pt", device="cpu") as handle:
            embedded_meta = handle.metadata() or {}
            _require(embedded_meta.get("model_id") == MODEL_ID
                     and embedded_meta.get("source_checkpoint_sha256") == CHECKPOINT_SHA256,
                     "Safetensors embedded model identity mismatch")
        tensors = load_file(str(weights), device="cpu")
    except (SafetensorError, OSError) as exc:
        raise ValueError('Cannot read safe model weights; obtain an intact trusted bundle') from exc
    expected_keys = set(_SCALERS) | {
        f"member_{member:03d}.{name}" for member in range(300) for name in _STATE_SHAPES
    }
    _require(set(tensors) == expected_keys, "Model weights have missing or unexpected keys")
    states = []
    for member in range(300):
        state = {name: tensors[f"member_{member:03d}.{name}"] for name in _STATE_SHAPES}
        _validate_state(state, member, torch)
        states.append(state)
    scalers = _validate_scalers(tensors)
    _require({name: str(value.dtype) for name, value in scalers.items()} == metadata["scaler_dtypes"],
             "Scaler dtypes do not match the manifest")
    _require(sha256(weights) == expected_sha, "Model weights changed during loading")
    return Ensemble(metadata=metadata, device=selected_device, _torch=torch,
                    _network=_build_network(torch, selected_device),
                    _states=states, _scalers=scalers)


def predict_embeddings(model: Ensemble, X: Any, batch_size: int = 256):
    """Return float64 ensemble scores for unscaled, finite N x 1280 embeddings.

    A private float32 copy is standardized in-place, preserving callers' arrays
    and the historical sklearn StandardScaler rounding. No parameters are fitted.
    """
    import numpy as np

    _require(isinstance(model, Ensemble), "Expected a model returned by load_model")
    _require(isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0,
             "Batch size must be a positive integer")
    values = np.asarray(X)
    _require(values.dtype.kind in "fiu", "Embeddings must be real numeric values")
    _require(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] == 1280,
             "Expected nonempty N x 1280 embeddings")
    raw = np.array(values, dtype=np.float32, copy=True)
    _require(bool(np.isfinite(raw).all()), "Embeddings must be finite float32 values")
    raw -= model._scalers["scaler_mean"]
    raw /= model._scalers["scaler_scale"]
    _require(bool(np.isfinite(raw).all()), "Standardized embeddings are non-finite")
    torch = model._torch
    torch.set_num_threads(2)
    model._network.eval()
    total = np.zeros(len(raw), dtype=np.float64)
    with torch.no_grad():
        for state in model._states:
            model._network.load_state_dict(state, strict=True)
            for start in range(0, len(raw), batch_size):
                inputs = torch.tensor(raw[start:start + batch_size], dtype=torch.float32, device=model.device)
                logits = model._network(inputs).cpu().numpy()
                # Match the frozen runtime exactly, including NumPy float32
                # sigmoid followed by accumulation into a float64 array.
                with np.errstate(over="ignore"):
                    total[start:start + len(logits)] += 1 / (1 + np.exp(-logits))
    scores = total / 300
    _require(bool(np.isfinite(scores).all()) and bool(((scores >= 0) & (scores <= 1)).all()),
             "Invalid model prediction scores")
    return scores
