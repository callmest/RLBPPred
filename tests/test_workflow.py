"""FASTA workflow and CLI contracts without networks or pretrained checkpoints.

Heavy structure/encoder/model operations are explicit software stubs. FASTA
reading, final CSV validation, receipts and command dispatch run normally.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rlbppred import __version__, cli, encoder, runtime, structure, workflow


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">sp|P49917|LIG4_HUMAN description\nAc\n>P38935 another description\nDE\n")
    assets = tmp_path / "assets"
    assets.mkdir()
    checkpoint = assets / "SaProt_650M_AF2.pt"
    checkpoint.write_bytes(b"software fixture; not a model checkpoint")
    foldseek = assets / "foldseek/bin/foldseek"
    foldseek.parent.mkdir(parents=True)
    foldseek.write_bytes(b"software fixture; never executed")
    foldseek.chmod(0o755)
    bundle = tmp_path / "model"
    bundle.mkdir()
    (bundle / "model.json").write_text("{}")
    (bundle / "model.safetensors").write_bytes(b"software fixture; never loaded")
    calls = []
    expected_records = [("P49917", "AC"), ("P38935", "DE")]
    expected_tokens = [("P49917", "ApCp"), ("P38935", "DpEp")]

    def resolve(records, workdir, **kwargs):
        assert records == expected_records
        calls.append(("resolve", Path(workdir), kwargs))
        return [{"protein_id": protein, "sequence": sequence} for protein, sequence in records]

    def tokens(resolved, binary, workdir, **kwargs):
        assert [item["protein_id"] for item in resolved] == ["P49917", "P38935"]
        assert Path(binary) == foldseek
        assert kwargs["plddt_threshold"] == 70.0
        calls.append(("tokens", Path(workdir), kwargs))
        return expected_tokens, {"foldseek_version": "10-test"}

    def encode(records, model, **kwargs):
        assert records == expected_tokens
        assert Path(model) == checkpoint
        calls.append(("encode", Path(model), kwargs))
        ids = [protein for protein, _ in records]
        matrix = np.arange(2560, dtype=np.float32).reshape(2, 1280)
        with Path(kwargs["output"]).open("xb") as target:
            np.savez_compressed(target, protein_id=np.array(ids), X=matrix)
        return ids, matrix, {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}

    model_sentinel = object()

    def load(model_dir, **kwargs):
        assert Path(model_dir) == bundle
        calls.append(("load", Path(model_dir), kwargs))
        return model_sentinel

    def predict(model, matrix, **kwargs):
        assert model is model_sentinel
        assert matrix.dtype == np.float32
        assert matrix.shape == (2, 1280)
        np.testing.assert_array_equal(matrix[1], np.arange(1280, 2560, dtype=np.float32))
        calls.append(("predict", matrix.shape, kwargs))
        return np.array([0.98, 0.40], dtype=np.float64)

    monkeypatch.setattr(structure, "resolve_structures", resolve)
    monkeypatch.setattr(structure, "make_tokens", tokens)
    monkeypatch.setattr(encoder, "encode_tokens", encode)
    monkeypatch.setattr(runtime, "load_model", load)
    monkeypatch.setattr(runtime, "predict_embeddings", predict)
    # A regression must never accidentally turn a default test into a network
    # request, even if the workflow changes its orchestration.
    monkeypatch.setattr(structure, "_fetch_bytes", lambda *a, **k: pytest.fail("network forbidden in unit tests"))
    return {"fasta": fasta, "assets": assets, "checkpoint": checkpoint, "foldseek": foldseek,
            "bundle": bundle, "calls": calls, "output": tmp_path / "run"}


def run_pipeline(pipeline, **options):
    kwargs = {"assets_dir": pipeline["assets"], "model_dir": pipeline["bundle"]}
    kwargs.update(options)
    return workflow.predict_fasta(pipeline["fasta"], pipeline["output"], **kwargs)


def test_fasta_to_prediction_preserves_order_and_writes_complete_receipt(pipeline):
    original = pipeline["fasta"].read_bytes()
    receipt = run_pipeline(pipeline, offline=True, structure_dir=pipeline["fasta"].parent,
                           structure_source="local", device="cpu", batch_size=7, max_residues=10)
    assert [call[0] for call in pipeline["calls"]] == ["resolve", "tokens", "encode", "load", "predict"]
    assert pipeline["calls"][0][2] == {"structure_dir": pipeline["fasta"].parent, "source": "local", "offline": True}
    assert pipeline["calls"][2][2]["max_residues"] == 10
    assert pipeline["calls"][4][2]["batch_size"] == 7
    with (pipeline["output"] / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["protein_id"] for row in rows] == ["P49917", "P38935"]
    assert [float(row["probability"]) for row in rows] == [0.98, 0.4]
    assert [row["above_screening_threshold"] for row in rows] == ["1", "0"]
    assert receipt == json.loads((pipeline["output"] / "run.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["rlbppred_version"] == __version__
    assert receipt["proteins"] == 2
    assert receipt["fasta_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["above_threshold"] == 1
    assert receipt["screening_threshold"] == 0.98
    assert receipt["operator"] == ">="
    assert receipt["foldseek_version"] == "10-test"
    assert receipt["model_id"] == runtime.MODEL_ID
    assert receipt["started_utc"] <= receipt["finished_utc"]
    assert "not automatically novel candidates" in receipt["note"]
    assert pipeline["fasta"].read_bytes() == original
    with np.load(pipeline["output"] / "features.npz", allow_pickle=False) as features:
        assert features["protein_id"].tolist() == ["P49917", "P38935"]
        assert features["X"].shape == (2, 1280)


@pytest.mark.parametrize("options", [{"batch_size": 0}, {"batch_size": -1}, {"batch_size": True},
                                      {"batch_size": 1.5}, {"threshold": float("nan")},
                                      {"threshold": float("inf")}, {"threshold": -0.1}, {"threshold": 1.1},
                                      {"max_residues": 0}, {"max_residues": True},
                                      {"max_residues": 1.5}, {"max_residues": 1}])
def test_invalid_parameters_fail_before_output_or_pipeline(pipeline, options):
    with pytest.raises(ValueError):
        run_pipeline(pipeline, **options)
    assert not pipeline["output"].exists()
    assert pipeline["calls"] == []


@pytest.mark.parametrize("relative", ["encoder", "foldseek", "manifest", "weights"])
def test_missing_asset_does_not_create_run_directory(pipeline, relative):
    paths = {"encoder": pipeline["checkpoint"], "foldseek": pipeline["foldseek"],
             "manifest": pipeline["bundle"] / "model.json", "weights": pipeline["bundle"] / "model.safetensors"}
    paths[relative].unlink()
    with pytest.raises(ValueError, match="not found"):
        run_pipeline(pipeline)
    assert not pipeline["output"].exists()
    assert pipeline["calls"] == []


def test_existing_output_directory_is_not_overwritten(pipeline):
    pipeline["output"].mkdir()
    original_receipt = b'{"status":"original"}\n'
    (pipeline["output"] / "run.json").write_bytes(original_receipt)
    with pytest.raises(FileExistsError, match="already exists"):
        run_pipeline(pipeline)
    assert (pipeline["output"] / "run.json").read_bytes() == original_receipt
    assert pipeline["calls"] == []


@pytest.mark.parametrize("target_exists", [True, False])
def test_symlink_output_directory_is_rejected(pipeline, tmp_path, target_exists):
    target = tmp_path / "linked-target"
    if target_exists:
        target.mkdir()
    pipeline["output"].symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="already exists"):
        run_pipeline(pipeline)
    assert pipeline["output"].is_symlink()
    assert pipeline["calls"] == []
    assert not (target / "run.json").exists()


def test_failed_structure_records_failure_and_stops_without_overwriting(pipeline, monkeypatch):
    original = pipeline["fasta"].read_bytes()

    def fail(*args, **kwargs):
        pipeline["calls"].append(("structure_failed", None, kwargs))
        raise ValueError("No complete structure chain exactly matches FASTA")

    monkeypatch.setattr(structure, "resolve_structures", fail)
    with pytest.raises(ValueError, match="No complete structure chain"):
        run_pipeline(pipeline)
    receipt_path = pipeline["output"] / "run.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "failed"
    assert receipt["error"] == "ValueError: No complete structure chain exactly matches FASTA"
    assert [call[0] for call in pipeline["calls"]] == ["structure_failed"]
    assert not (pipeline["output"] / "predictions.csv").exists()
    assert not (pipeline["output"] / "features.npz").exists()
    before_retry = receipt_path.read_bytes()
    with pytest.raises(FileExistsError):
        run_pipeline(pipeline)
    assert receipt_path.read_bytes() == before_retry
    assert pipeline["fasta"].read_bytes() == original


def test_failed_encoder_does_not_load_mlp_or_write_csv(pipeline, monkeypatch):
    def fail(*args, **kwargs):
        pipeline["calls"].append(("encoder_failed", None, kwargs))
        raise RuntimeError("Checkpoint identity mismatch")

    monkeypatch.setattr(encoder, "encode_tokens", fail)
    with pytest.raises(RuntimeError, match="Checkpoint identity"):
        run_pipeline(pipeline)
    assert [call[0] for call in pipeline["calls"]] == ["resolve", "tokens", "encoder_failed"]
    assert json.loads((pipeline["output"] / "run.json").read_text())["status"] == "failed"
    assert not (pipeline["output"] / "predictions.csv").exists()


def test_csv_validation_failure_is_not_reported_as_success(pipeline, monkeypatch):
    monkeypatch.setattr(runtime, "predict_embeddings", lambda *a, **k: np.array([0.9, float("nan")]))
    with pytest.raises(ValueError, match="finite score"):
        run_pipeline(pipeline)
    receipt = json.loads((pipeline["output"] / "run.json").read_text())
    assert receipt["status"] == "failed"
    assert "above_threshold" not in receipt
    assert not (pipeline["output"] / "predictions.csv").exists()


def test_predict_fasta_cli_runs_the_same_pipeline(pipeline, capsys):
    code = cli.main(["predict-fasta", "--fasta", str(pipeline["fasta"]),
                     "--output-dir", str(pipeline["output"]), "--assets-dir", str(pipeline["assets"]),
                     "--model-dir", str(pipeline["bundle"]), "--offline", "--structure-source", "local",
                     "--structure-dir", str(pipeline["fasta"].parent), "--batch-size", "11",
                     "--max-residues", "12", "--threshold", "0.99"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "complete"
    assert printed["screening_threshold"] == 0.99
    assert printed["above_threshold"] == 0
    assert pipeline["calls"][-1][2]["batch_size"] == 11
    assert pipeline["calls"][2][2]["max_residues"] == 12
    assert json.loads((pipeline["output"] / "run.json").read_text()) == printed


@pytest.mark.parametrize("arguments", [["--batch-size", "0"], ["--threshold", "nan"],
                                        ["--threshold", "1.01"], ["--max-residues", "1"]])
def test_cli_invalid_fasta_settings_return_clean_failure(pipeline, capsys, arguments):
    code = cli.main(["predict-fasta", "--fasta", str(pipeline["fasta"]),
                     "--output-dir", str(pipeline["output"]), "--assets-dir", str(pipeline["assets"]),
                     "--model-dir", str(pipeline["bundle"]), *arguments])
    assert code == 2
    assert "rlbppred:" in capsys.readouterr().err
    assert not pipeline["output"].exists()
    assert pipeline["calls"] == []


def test_cli_version_is_current(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "rlbppred 0.2.1"


def test_setup_model_dispatch_uses_requested_asset_cache(tmp_path, monkeypatch, capsys):
    from rlbppred import assets

    asset_cache = tmp_path / "requested-assets"
    model_dir = tmp_path / "requested-model"
    calls = []

    def download_model(destination, *, cache_dir):
        calls.append((destination, cache_dir))
        return destination

    monkeypatch.setattr(assets, "download_model", download_model)
    monkeypatch.setattr(assets, "download_saprot", lambda *a, **k: pytest.fail("model-only setup fetched encoder"))
    monkeypatch.setattr(assets, "download_foldseek", lambda *a, **k: pytest.fail("model-only setup fetched Foldseek"))
    code = cli.main(["setup", "--components", "model", "--assets-dir", str(asset_cache),
                     "--model-dir", str(model_dir)])
    assert code == 0
    assert calls == [(model_dir, asset_cache)]
    assert json.loads(capsys.readouterr().out) == {"model": str(model_dir)}
    assert not asset_cache.exists()
    assert not model_dir.exists()
