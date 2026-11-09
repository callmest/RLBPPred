"""Encoder contract tests use tiny models, never upstream model downloads."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rlbppred import encoder


class TinyAlphabet:
    def __init__(self, standard_toks, **kwargs):
        assert kwargs == {"prepend_toks": [], "append_toks": [], "prepend_bos": True,
                          "append_eos": True, "use_msa": False}
        self.all_toks = list(standard_toks) + ["<null_1>", "<null_2>"]
        self.cls_idx, self.padding_idx, self.eos_idx = 0, 1, 2
        self.unk_idx, self.mask_idx = 3, 4


class TinyESM2(torch.nn.Module):
    instances = []

    def __init__(self, alphabet, **config):
        super().__init__()
        assert config == encoder._CONFIG
        self.num_layers = config["num_layers"]
        self.alphabet = alphabet
        self.register_buffer("known", torch.tensor(1.0))
        self.seen = []
        self.__class__.instances.append(self)

    def forward(self, tokens, repr_layers):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert repr_layers == [33]
        self.seen.append(tokens.detach().cpu().tolist()[0])
        # Large special-token values make accidental BOS/EOS pooling obvious.
        value = tokens.to(torch.float32) * self.known
        value[:, 0] = 10000
        value[:, -1] = 20000
        return {"representations": {33: value.unsqueeze(-1).expand(-1, -1, 1280)}}


@pytest.fixture
def tiny_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "SaProt_650M_AF2.pt"
    torch.save({"model": {"known": torch.tensor(1.0)}, "config": dict(encoder._CONFIG)}, path)
    monkeypatch.setattr(encoder, "CHECKPOINT_BYTES", path.stat().st_size)
    monkeypatch.setattr(encoder, "CHECKPOINT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(encoder, "_esm_backend", lambda: (TinyAlphabet, TinyESM2))
    TinyESM2.instances = []
    return path


def test_exact_alphabet_and_final_mean(tiny_checkpoint, tmp_path):
    output = tmp_path / "embeddings.npz"
    ids, matrix, metadata = encoder.encode_tokens([("P1", "ApCy"), ("P2", "##")],
                                                 tiny_checkpoint, output=output)
    assert ids == ["P1", "P2"]
    assert matrix.dtype == np.float32 and matrix.shape == (2, 1280)
    # Ap=5, Cy=27 and ##=445 in the original 446-entry alphabet.
    np.testing.assert_array_equal(matrix[0], np.full(1280, 16, dtype=np.float32))
    np.testing.assert_array_equal(matrix[1], np.full(1280, 445, dtype=np.float32))
    assert TinyESM2.instances[0].seen == [[0, 5, 27, 2], [0, 445, 2]]
    assert metadata["representation_layer"] == 33
    assert metadata["residue_counts"] == [2, 1]
    assert not metadata["truncated"] and not metadata["chunked"]
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["X"], matrix)
        assert data["protein_id"].tolist() == ids
        assert json.loads(data["metadata"].item()) == metadata


@pytest.mark.parametrize("records", [[], [("P1", "A")], [("P1", "Apxy")],
                                   [("P1", "AP")], [("P1", "<unk>")],
                                   [("P1", "")], [("P1", "Ap"), ("P1", "Cy")],
                                   [("../P1", "Ap")], [(1, "Ap")], [("P1", "Ap", "x")]])
def test_invalid_records_fail_before_loading(records, monkeypatch):
    loader = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(encoder, "_load_checkpoint", loader)
    with pytest.raises(ValueError):
        encoder.encode_tokens(records, "nonexistent.pt")
    loader.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "10"])
def test_invalid_limit(limit):
    with pytest.raises(ValueError, match="max_residues"):
        encoder.encode_tokens([("P1", "Ap")], "missing", max_residues=limit)


def test_length_limit_is_explicit_and_no_1024_truncation(tiny_checkpoint):
    with pytest.raises(ValueError, match="not truncated or chunked"):
        encoder.encode_tokens([("P1", "Ap" * 3)], tiny_checkpoint, max_residues=2)
    ids, matrix, metadata = encoder.encode_tokens([("P1", "Ap" * 1100)], tiny_checkpoint)
    assert ids == ["P1"] and matrix.shape == (1, 1280)
    assert len(TinyESM2.instances[-1].seen[0]) == 1102
    assert metadata["residue_counts"] == [1100]


@pytest.mark.parametrize("problem", ["hash", "size"])
def test_authentication_precedes_torch_load(tiny_checkpoint, monkeypatch, problem):
    if problem == "hash":
        monkeypatch.setattr(encoder, "CHECKPOINT_SHA256", "0" * 64)
    else:
        monkeypatch.setattr(encoder, "CHECKPOINT_BYTES", 99)
    load = Mock(side_effect=AssertionError("must not deserialize"))
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(ValueError, match="refusing deserialization"):
        encoder.encode_tokens([("P1", "Ap")], tiny_checkpoint)
    load.assert_not_called()


def test_safe_load_no_pickle_fallback(tiny_checkpoint, monkeypatch):
    load = Mock(side_effect=RuntimeError("restricted loader refused object"))
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(ValueError, match="arbitrary pickle fallback is not permitted"):
        encoder.encode_tokens([("P1", "Ap")], tiny_checkpoint)
    assert load.call_count == 1
    assert load.call_args.kwargs == {"map_location": "cpu", "weights_only": True}


@pytest.mark.parametrize("weights", [{}, {"unexpected": torch.tensor(1.0)},
                                    {"known": torch.tensor(1.0), "extra": torch.tensor(2.0)}])
def test_missing_or_extra_weights_fail_strictly(tiny_checkpoint, monkeypatch, weights):
    # Authentication is covered separately; this isolates strict model loading.
    monkeypatch.setattr(encoder, "_load_checkpoint", lambda checkpoint, backend: weights)
    with pytest.raises(ValueError, match="strictly match"):
        encoder.encode_tokens([("P1", "Ap")], tiny_checkpoint)


def test_only_unused_contact_prediction_head_can_be_removed(tiny_checkpoint, monkeypatch):
    class WithContactHead(TinyESM2):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.contact_head = torch.nn.Module()
            self.contact_head.regression = torch.nn.Linear(2, 1)
    monkeypatch.setattr(encoder, "_esm_backend", lambda: (TinyAlphabet, WithContactHead))
    _, matrix, metadata = encoder.encode_tokens([("P1", "Ap")], tiny_checkpoint)
    assert metadata["omitted_contact_prediction_head"] is True
    assert WithContactHead.instances[-1].contact_head is None
    np.testing.assert_array_equal(matrix, np.full((1, 1280), 5, dtype=np.float32))
    # Missing encoder parameters are never excused by the optional contact head.
    monkeypatch.setattr(encoder, "_load_checkpoint", lambda checkpoint, backend: {})
    with pytest.raises(ValueError, match="strictly match"):
        encoder.encode_tokens([("P1", "Ap")], tiny_checkpoint)


@pytest.mark.parametrize("device", ["mps", "auto", "cuda:x", "cuda:-1", "CUDA", None])
def test_invalid_device_fails(tiny_checkpoint, device):
    with pytest.raises(ValueError, match="Device"):
        encoder.encode_tokens([("P1", "Ap")], tiny_checkpoint, device=device)


def test_cuda_unavailable_never_falls_back(tiny_checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        encoder.encode_tokens([("P1", "Ap")], tiny_checkpoint, device="cuda")
    assert not TinyESM2.instances


def test_existing_and_symlink_outputs_are_preserved(tiny_checkpoint, tmp_path):
    existing = tmp_path / "result.npz"
    existing.write_bytes(b"user-data")
    link = tmp_path / "link.npz"
    link.symlink_to(tmp_path / "missing")
    for output in (existing, link):
        with pytest.raises(FileExistsError):
            encoder.encode_tokens([("P1", "Ap")], tiny_checkpoint, output=output)
    assert existing.read_bytes() == b"user-data" and link.is_symlink()
    assert not TinyESM2.instances


def test_extraction_failure_does_not_publish_partial_file(tiny_checkpoint, tmp_path, monkeypatch):
    class BadESM2(TinyESM2):
        def forward(self, tokens, repr_layers):
            result = super().forward(tokens, repr_layers)
            result["representations"][33] = torch.full((1, tokens.shape[1], 1280), float("nan"))
            return result
    monkeypatch.setattr(encoder, "_esm_backend", lambda: (TinyAlphabet, BadESM2))
    output = tmp_path / "result.npz"
    with pytest.raises(ValueError, match="non-finite"):
        encoder.encode_tokens([("P1", "Ap")], tiny_checkpoint, output=output)
    assert not output.exists()


def test_atomic_output_rejects_concurrent_file(tmp_path):
    output = tmp_path / "result.npz"
    output.write_bytes(b"concurrent-user-data")
    with pytest.raises(FileExistsError):
        encoder._write_npz(output, ["P1"], np.zeros((1, 1280), np.float32), {})
    assert output.read_bytes() == b"concurrent-user-data"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.npz"]


def test_backend_rejects_wrong_version(monkeypatch):
    monkeypatch.setattr(encoder.importlib.metadata, "distribution", lambda name: SimpleNamespace(version="3.0.0"))
    with pytest.raises(RuntimeError, match="exactly fair-esm"):
        encoder._esm_backend()


def test_backend_rejects_shadowing_esm(tmp_path, monkeypatch):
    distribution = SimpleNamespace(version="2.0.0", locate_file=lambda relative: tmp_path / relative)
    monkeypatch.setattr(encoder.importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(encoder.importlib, "import_module", lambda name: SimpleNamespace(__file__="/unrelated/esm.py"))
    with pytest.raises(RuntimeError, match="do not belong"):
        encoder._esm_backend()


def test_backend_accepts_expected_fair_esm_paths(tmp_path, monkeypatch):
    distribution = SimpleNamespace(version="2.0.0", locate_file=lambda relative: tmp_path / relative)
    modules = {
        "esm": SimpleNamespace(__file__=str(tmp_path / "esm/__init__.py")),
        "esm.data": SimpleNamespace(__file__=str(tmp_path / "esm/data.py"), Alphabet=TinyAlphabet),
        "esm.model.esm2": SimpleNamespace(__file__=str(tmp_path / "esm/model/esm2.py"), ESM2=TinyESM2),
    }
    monkeypatch.setattr(encoder.importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(encoder.importlib, "import_module", lambda name: modules[name])
    assert encoder._esm_backend() == (TinyAlphabet, TinyESM2)
