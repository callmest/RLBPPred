"""Release identity, deserialization, device and numerical-path regression guards."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rlbppred import cli, runtime


@pytest.fixture
def manifest():
    return {
        "schema_version": 1,
        "model_id": runtime.MODEL_ID,
        "format": "safetensors_ensemble_v1",
        "source_checkpoint_sha256": runtime.CHECKPOINT_SHA256,
        "upstream": "saprot-650m-af2",
        "classifier": "MLP",
        "seed": 42,
        "architecture": deepcopy(runtime._ARCHITECTURE),
        "preprocessing": deepcopy(runtime._PREPROCESSING),
        "scaler_dtypes": {name: "float64" for name in runtime._SCALERS},
        "weights": {"filename": "model.safetensors", "sha256": "a" * 64},
        "classification_threshold": 0.5,
        "screening_threshold": 0.98,
    }


def forbid_framework():
    raise AssertionError("Deserialization/framework import must not be reached")


def test_unknown_checkpoint_is_rejected_before_framework_or_pickle(tmp_path, monkeypatch):
    checkpoint = tmp_path / "not-trusted.pt"
    checkpoint.write_bytes(b"This is deliberately not the trusted checkpoint.")
    output = tmp_path / "model"
    monkeypatch.setattr(runtime, "_torch", forbid_framework)
    monkeypatch.setattr(runtime, "_safetensors", forbid_framework)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime.export_model(checkpoint, output)
    assert not output.exists()
    assert checkpoint.read_bytes() == b"This is deliberately not the trusted checkpoint."


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_export_never_overwrites_an_existing_destination(tmp_path, monkeypatch, kind):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"source")
    output = tmp_path / "existing"
    if kind == "directory":
        output.mkdir()
        protected = output / "keep.txt"
    else:
        protected = output
    protected.write_bytes(b"keep this unchanged")
    monkeypatch.setattr(runtime, "_torch", forbid_framework)
    with pytest.raises(ValueError, match="Output already exists"):
        runtime.export_model(checkpoint, output)
    assert protected.read_bytes() == b"keep this unchanged"


def test_export_rejects_existing_dangling_symlink_before_hashing(tmp_path, monkeypatch):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"source")
    target = tmp_path / "not-created"
    link = tmp_path / "already-existing-link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(runtime, "sha256", lambda _: (_ for _ in ()).throw(
        AssertionError("A pre-existing destination must be rejected before reading the checkpoint")))
    with pytest.raises(ValueError, match="(?i)(exist|symlink)"):
        runtime.export_model(checkpoint, link)
    assert link.is_symlink() and not target.exists()


@pytest.mark.parametrize("filename", [
    "../model.safetensors", "/tmp/model.safetensors", "nested/model.safetensors",
    "..", ".", "model.safetensors/../outside", r"..\model.safetensors", "model.pt",
])
def test_manifest_refuses_nonflat_or_nonrelease_weights(manifest, filename):
    manifest["weights"]["filename"] = filename
    with pytest.raises(ValueError, match="flat filename"):
        runtime._validate_manifest(manifest)


@pytest.mark.parametrize("key,value", [
    ("schema_version", 2), ("model_id", "another-model"),
    ("format", "pickle"), ("source_checkpoint_sha256", "0" * 64),
    ("upstream", "saprot-650m-pdb"), ("classifier", "SVM"),
    ("seed", 1), ("classification_threshold", 0.98), ("screening_threshold", 0.85),
])
def test_manifest_identity_is_pinned(manifest, key, value):
    manifest[key] = value
    with pytest.raises(ValueError, match="manifest mismatch"):
        runtime._validate_manifest(manifest)


@pytest.mark.parametrize("field,value", [
    ("input_dim", 640), ("hidden_dims", [256, 128]), ("dropout", 0.1),
    ("batch_norm", False), ("members", 299),
])
def test_manifest_rejects_wrong_architecture(manifest, field, value):
    manifest["architecture"][field] = value
    with pytest.raises(ValueError, match="architecture"):
        runtime._validate_manifest(manifest)


@pytest.mark.parametrize("digest", [None, 1, "", "a" * 63, "a" * 65, "G" * 64, "A" * 64])
def test_manifest_rejects_invalid_weight_hash(manifest, digest):
    manifest["weights"]["sha256"] = digest
    with pytest.raises(ValueError, match="SHA-256"):
        runtime._validate_manifest(manifest)


@pytest.mark.parametrize("dtypes", [None, {}, {"scaler_mean": "float64"},
    {name: "float16" for name in runtime._SCALERS},
    {name: "object" for name in runtime._SCALERS},
    {**{name: "float64" for name in runtime._SCALERS}, "extra": "float64"},
])
def test_manifest_rejects_bad_scaler_dtypes(manifest, dtypes):
    manifest["scaler_dtypes"] = dtypes
    with pytest.raises(ValueError, match="dtype"):
        runtime._validate_manifest(manifest)


def test_valid_manifest_round_trips_without_frameworks(manifest):
    assert runtime._validate_manifest(json.loads(json.dumps(manifest))) == (
        "model.safetensors", "a" * 64)


def make_fake_model():
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeTorch:
        float32 = np.float32
        thread_counts = []

        @classmethod
        def set_num_threads(cls, n):
            cls.thread_counts.append(n)

        @staticmethod
        def no_grad():
            return nullcontext()

        @staticmethod
        def tensor(values, dtype, device):
            assert dtype == np.float32 and device == "cpu"
            return np.array(values, dtype=dtype, copy=True)

    class FakeNetwork:
        def __init__(self):
            self.loaded = []
            self.evaluated = False

        def eval(self):
            self.evaluated = True
            return self

        def load_state_dict(self, state, strict):
            assert strict
            self.state = state
            self.loaded.append(state)

        def __call__(self, values):
            assert self.evaluated
            logits = values[:, 0] + np.float32(self.state["offset"])
            return FakeTensor(np.asarray(logits, dtype=np.float32))

    return runtime.Ensemble(
        metadata={}, device="cpu", _torch=FakeTorch, _network=FakeNetwork(),
        _states=[{"offset": i / 1000} for i in range(300)],
        _scalers={"scaler_mean": np.full(1280, .123456789, dtype=np.float64),
                  "scaler_scale": np.full(1280, 1.23456789, dtype=np.float64),
                  "scaler_var": np.full(1280, 1.524, dtype=np.float64)},
    )


@pytest.mark.parametrize("array", [
    np.zeros((0, 1280), dtype=np.float32), np.zeros(1280, dtype=np.float32),
    np.zeros((2, 1279), dtype=np.float32), np.zeros((2, 1280, 1), dtype=np.float32),
    np.full((1, 1280), np.nan, dtype=np.float32),
    np.full((1, 1280), np.inf, dtype=np.float32),
    np.full((1, 1280), -np.inf, dtype=np.float32),
    np.full((1, 1280), 1e300, dtype=np.float64),
    np.zeros((1, 1280), dtype=complex), np.zeros((1, 1280), dtype=bool),
    np.full((1, 1280), "not numeric"), np.full((1, 1280), object(), dtype=object),
])
def test_prediction_rejects_invalid_input_before_network_use(array):
    model = make_fake_model()
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
        runtime.predict_embeddings(model, array)
    assert not model._network.loaded


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.5, "256", None])
def test_prediction_rejects_invalid_batch_size(batch_size):
    model = make_fake_model()
    with pytest.raises(ValueError, match="positive integer"):
        runtime.predict_embeddings(model, np.zeros((1, 1280), dtype=np.float32), batch_size)
    assert not model._network.loaded


def test_prediction_requires_an_ensemble_instance():
    with pytest.raises(ValueError, match="load_model"):
        runtime.predict_embeddings(None, np.zeros((1, 1280), dtype=np.float32))


def test_prediction_keeps_original_float32_scaling_sigmoid_and_float64_average():
    model = make_fake_model()
    X = np.linspace(-3, 3, 5 * 1280, dtype=np.float32).reshape(5, 1280)
    original = X.copy()
    scaled = X.copy()
    scaled -= model._scalers["scaler_mean"]
    scaled /= model._scalers["scaler_scale"]
    expected = np.zeros(5, dtype=np.float64)
    for state in model._states:
        logits = scaled[:, 0] + np.float32(state["offset"])
        expected += 1 / (1 + np.exp(-logits))
    expected /= 300
    predicted = runtime.predict_embeddings(model, X, batch_size=2)
    np.testing.assert_array_equal(predicted, expected)
    np.testing.assert_array_equal(X, original)
    assert predicted.dtype == np.dtype("float64")
    assert len(model._network.loaded) == 300
    assert model._torch.thread_counts[-1] == 2


def fake_cuda(available=True, count=2):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available,
                                              device_count=lambda: count))


@pytest.mark.parametrize("device", ["gpu", "mps", "xpu", "cuda:-1", "cuda:1.5", " cpu", "CPU", None, 1])
def test_invalid_devices_are_rejected(device):
    with pytest.raises(ValueError):
        runtime._device(fake_cuda(), device)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
def test_unavailable_cuda_does_not_fall_back_to_cpu(device):
    with pytest.raises(ValueError, match="unavailable"):
        runtime._device(fake_cuda(available=False), device)


def test_cuda_index_is_checked_and_explicit_device_preserved():
    with pytest.raises(ValueError, match="index 2"):
        runtime._device(fake_cuda(count=2), "cuda:2")
    assert runtime._device(fake_cuda(), "cpu") == "cpu"
    assert runtime._device(fake_cuda(), "cuda") == "cuda"
    assert runtime._device(fake_cuda(), "cuda:1") == "cuda:1"


def test_runtime_import_does_not_import_frameworks_or_numpy(tmp_path):
    script = (
        "import sys; import rlbppred.runtime; "
        "assert 'torch' not in sys.modules; "
        "assert 'safetensors' not in sys.modules; "
        "assert 'numpy' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_corrupt_safetensors_has_a_clean_cli_error(tmp_path, manifest, capsys):
    model_dir = tmp_path / 'bad-model'
    model_dir.mkdir()
    weights = model_dir / 'model.safetensors'
    weights.write_bytes(b'not a safetensors header')
    manifest['weights']['sha256'] = runtime.sha256(weights)
    (model_dir / 'model.json').write_text(json.dumps(manifest))
    assert cli.main(['model', 'verify', '--model-dir', str(model_dir)]) == 2
    output = capsys.readouterr()
    assert 'Cannot read safe model weights' in output.err
    assert 'Traceback' not in output.err


def test_corrupt_tensor_has_a_clean_pack_error(tmp_path, capsys):
    directory = tmp_path / 'features'
    directory.mkdir()
    (directory / 'TEST.pt').write_bytes(b'not a tensor archive')
    ids = tmp_path / 'ids.txt'
    ids.write_text('TEST\n')
    output = tmp_path / 'features.npz'
    assert cli.main(['pack', '--embedding-dir', str(directory), '--ids', str(ids), '--output', str(output)]) == 2
    captured = capsys.readouterr()
    assert 'arbitrary pickle is not accepted' in captured.err
    assert 'Traceback' not in captured.err
    assert not output.exists()


def test_cli_model_install_refuses_existing_destination_before_loading(tmp_path, monkeypatch, capsys):
    output = tmp_path / "model"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_bytes(b"preserve")
    monkeypatch.setattr(runtime, "load_model", lambda *_a, **_k: forbid_framework())
    assert cli.main(["model", "install", "--source", str(tmp_path / "source"),
                     "--model-dir", str(output)]) == 2
    assert keep.read_bytes() == b"preserve"
    assert "already exists" in capsys.readouterr().err


def test_cli_install_copies_only_two_verified_bundle_files(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.json").write_text('{"example": true}')
    (source / "model.safetensors").write_bytes(b"example fixture")
    (source / "never-copy.py").write_text('raise RuntimeError("never execute")')
    calls = []

    def load(path, **kwargs):
        calls.append(Path(path))
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(runtime, "load_model", load)
    target = tmp_path / "cache" / "bundle"
    assert cli.main(["model", "install", "--source", str(source), "--model-dir", str(target)]) == 0
    assert calls == [source, target]
    assert sorted(p.name for p in target.iterdir()) == ["model.json", "model.safetensors"]
    assert (target / "model.safetensors").read_bytes() == b"example fixture"
    assert (source / "never-copy.py").is_file()


def test_cli_failed_install_removes_only_new_bundle(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.json").write_text('{}')
    (source / "model.safetensors").write_bytes(b"fixture")
    protected = tmp_path / "unrelated.txt"
    protected.write_bytes(b"do not change")
    target = tmp_path / "target"

    def load(path, **kwargs):
        if Path(path) == target:
            raise ValueError("Post-copy verification failure")
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(runtime, "load_model", load)
    assert cli.main(["model", "install", "--source", str(source), "--model-dir", str(target)]) == 2
    assert not target.exists()
    assert (source / "model.safetensors").read_bytes() == b"fixture"
    assert protected.read_bytes() == b"do not change"


def test_load_refuses_hash_mismatch_before_framework_loading(tmp_path, monkeypatch, manifest):
    (tmp_path / "model.json").write_text(json.dumps(manifest))
    (tmp_path / "model.safetensors").write_bytes(b"bad hash")
    monkeypatch.setattr(runtime, "_torch", forbid_framework)
    with pytest.raises(ValueError, match="weights SHA-256 mismatch"):
        runtime.load_model(tmp_path)


@pytest.mark.parametrize("symlink_name", ["model.json", "model.safetensors"])
def test_load_refuses_bundle_symlinks(tmp_path, monkeypatch, manifest, symlink_name):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.json").write_text(json.dumps(manifest))
    (bundle / "model.safetensors").write_bytes(b"fixture")
    outside = tmp_path / (symlink_name + ".outside")
    (bundle / symlink_name).rename(outside)
    (bundle / symlink_name).symlink_to(outside)
    monkeypatch.setattr(runtime, "_torch", forbid_framework)
    with pytest.raises(ValueError, match="non-symlink"):
        runtime.load_model(bundle)
    assert outside.is_file()
