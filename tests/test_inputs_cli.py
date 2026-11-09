"""Input/output and installed-command contracts; no models or network needed.

These are software fixtures, not biological validation data. Raw, unscaled
SaProt-AF2 embeddings are the documented input; numeric checks cannot establish
the biological provenance of an arbitrary feature matrix.
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path

import numpy as np
import pytest

from rlbppred.io import load_npz, write_predictions


def save_features(path: Path, ids=None, X=None) -> Path:
    if ids is None:
        ids = np.array(["Q9ZZZ9", "P49917"], dtype="U16")
    if X is None:
        X = np.arange(2560, dtype=np.float32).reshape(2, 1280) / 7
    np.savez(path, protein_id=ids, X=X)
    return path


def invoke_main(args):
    """Accept either normal CLI exit codes or argparse's SystemExit."""
    cli = importlib.import_module("rlbppred.cli")
    try:
        result = cli.main(args)
    except SystemExit as exc:
        return exc.code or 0
    return result or 0


def test_load_valid_raw_float32_preserves_order_values_and_source(tmp_path):
    X = np.arange(2560, dtype=np.float32).reshape(2, 1280) / 7
    path = save_features(tmp_path / "features.npz", X=X)
    original = path.read_bytes()
    ids, actual = load_npz(path)
    assert ids == ["Q9ZZZ9", "P49917"]
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, X)
    assert path.read_bytes() == original


def test_valid_identifier_characters(tmp_path):
    ids = np.array(["P49917", "Q9H0A0-2", "toy:1|v.2+tag", "123", "A_B"], dtype="U32")
    path = save_features(tmp_path / "features.npz", ids=ids, X=np.zeros((5, 1280), np.float32))
    loaded, _ = load_npz(path)
    assert loaded == ids.tolist()


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.int32, np.uint8])
def test_non_float32_features_are_rejected(tmp_path, dtype):
    path = save_features(tmp_path / "features.npz", X=np.zeros((2, 1280), dtype=dtype))
    with pytest.raises((ValueError, TypeError)):
        load_npz(path)


@pytest.mark.parametrize("shape", [(2, 1279), (2, 1281), (1280,), (2, 1, 1280), (0, 1280)])
def test_wrong_feature_shapes_are_rejected(tmp_path, shape):
    path = save_features(tmp_path / "features.npz", X=np.zeros(shape, np.float32))
    with pytest.raises((ValueError, TypeError)):
        load_npz(path)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_features_are_rejected(tmp_path, value):
    X = np.zeros((2, 1280), np.float32)
    X[1, 1279] = value
    path = save_features(tmp_path / "features.npz", X=X)
    with pytest.raises((ValueError, TypeError)):
        load_npz(path)


@pytest.mark.parametrize(
    "ids",
    [
        np.array(["P49917", "P49917"]),
        np.array(["P49917"]),
        np.array([["P49917", "Q9ZZZ9"]]),
        np.array([], dtype="U1"),
        np.array([1, 2], dtype=np.int64),
    ],
)
def test_invalid_identifier_arrays_are_rejected(tmp_path, ids):
    path = save_features(tmp_path / "features.npz", ids=ids)
    with pytest.raises((ValueError, TypeError)):
        load_npz(path)


@pytest.mark.parametrize(
    "bad_id",
    ["", " ", " A", "A B", "A/B", "A\\B", "../A", "_A", "-A", "=SUM(A1)", "A\nB", "A\tB", "蛋白A"],
)
def test_unsafe_or_ambiguous_identifiers_are_rejected(tmp_path, bad_id):
    path = save_features(tmp_path / "features.npz", ids=np.array(["P49917", bad_id]))
    with pytest.raises((ValueError, TypeError)):
        load_npz(path)


@pytest.mark.parametrize("object_key", ["protein_id", "X"])
def test_object_arrays_never_enable_pickle(tmp_path, object_key):
    ids = np.array(["P49917", "Q9ZZZ9"])
    X = np.zeros((2, 1280), np.float32)
    if object_key == "protein_id":
        ids = ids.astype(object)
    else:
        X = X.astype(object)
    path = save_features(tmp_path / "features.npz", ids=ids, X=X)
    with pytest.raises((ValueError, TypeError)):
        load_npz(path)


@pytest.mark.parametrize("missing", ["protein_id", "X"])
def test_missing_required_arrays_are_rejected(tmp_path, missing):
    arrays = {"protein_id": np.array(["P49917"]), "X": np.zeros((1, 1280), np.float32)}
    del arrays[missing]
    path = tmp_path / "features.npz"
    np.savez(path, **arrays)
    with pytest.raises((ValueError, TypeError, KeyError)):
        load_npz(path)


def test_invalid_npz_is_rejected(tmp_path):
    path = tmp_path / "features.npz"
    path.write_text("not a NumPy archive", encoding="utf-8")
    with pytest.raises((ValueError, TypeError, OSError)):
        load_npz(path)


def test_prediction_csv_keeps_order_precision_and_inclusive_boundaries(tmp_path):
    output = tmp_path / "scores.csv"
    ids = ["Z9", "A1", "B2", "C3", "D4"]
    probs = np.array([0.9800000000001234, 0.98, 0.5, 0.4999999999999, 0.0], dtype=np.float64)
    write_predictions(output, ids, probs)
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "protein_id", "probability", "classification_at_0_5", "above_screening_threshold"
        ]
        rows = list(reader)
    assert [r["protein_id"] for r in rows] == ids
    assert [float(r["probability"]) for r in rows] == probs.tolist()
    assert [int(r["classification_at_0_5"]) for r in rows] == [1, 1, 1, 0, 0]
    assert [int(r["above_screening_threshold"]) for r in rows] == [1, 1, 0, 0, 0]


def test_prediction_csv_respects_explicit_screening_threshold(tmp_path):
    output = tmp_path / "scores.csv"
    write_predictions(output, ["A1", "B2"], np.array([0.75, 0.749999]), threshold=0.75)
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["above_screening_threshold"]) for r in rows] == [1, 0]


def test_prediction_output_is_exclusive_and_preserves_existing_content(tmp_path):
    output = tmp_path / "scores.csv"
    output.write_bytes(b"existing scientific result\n")
    with pytest.raises((ValueError, FileExistsError, OSError)):
        write_predictions(output, ["A1"], np.array([0.8]))
    assert output.read_bytes() == b"existing scientific result\n"


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_prediction_threshold_fails_without_output(tmp_path, threshold):
    output = tmp_path / "scores.csv"
    with pytest.raises((ValueError, TypeError)):
        write_predictions(output, ["A1"], np.array([0.8]), threshold=threshold)
    assert not output.exists()


@pytest.mark.parametrize("probabilities", [[np.nan], [np.inf], [-0.01], [1.01], []])
def test_invalid_probabilities_fail_without_output(tmp_path, probabilities):
    output = tmp_path / "scores.csv"
    with pytest.raises((ValueError, TypeError)):
        write_predictions(output, ["A1"], np.asarray(probabilities, dtype=np.float64))
    assert not output.exists()


@pytest.mark.parametrize("args", [["--help"], ["--version"], ["predict", "--help"]])
def test_help_and_version_are_available(args, capsys):
    assert invoke_main(args) == 0
    captured = capsys.readouterr()
    assert captured.out.strip()
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize(
    "args", [["predict"], ["predict", "--embeddings", "features.npz"], ["predict", "--output", "scores.csv"]]
)
def test_predict_requires_both_input_and_output(args, capsys):
    assert invoke_main(args) != 0
    captured = capsys.readouterr()
    assert captured.err.strip()
    assert "Traceback" not in captured.out + captured.err


def test_bad_input_fails_before_loading_model(tmp_path, monkeypatch, capsys):
    cli = importlib.import_module("rlbppred.cli")
    runtime = importlib.import_module("rlbppred.runtime")
    path = tmp_path / "invalid-features.npz"
    path.write_text("not an embedding archive", encoding="utf-8")
    output = tmp_path / "scores.csv"

    def forbidden_model_load(*args, **kwargs):
        pytest.fail("Invalid input must be rejected before loading the large model")

    monkeypatch.setattr(runtime, "load_model", forbidden_model_load)
    monkeypatch.setattr(cli, "load_model", forbidden_model_load, raising=False)
    assert invoke_main(["predict", "--embeddings", str(path), "--output", str(output)]) != 0
    captured = capsys.readouterr()
    assert captured.err.strip()
    assert "Traceback" not in captured.out + captured.err
    assert not output.exists()


def test_cli_help_exposes_embedding_input_and_screening_threshold(capsys):
    assert invoke_main(["predict", "--help"]) == 0
    help_text = capsys.readouterr().out.lower()
    assert "threshold" in help_text
    assert "embeddings" in help_text
