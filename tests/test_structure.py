"""Strict structure-preparation contracts using synthetic software fixtures.

No live services, real biological predictions or downloaded binaries are used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from rlbppred import structure


RESIDUES = {v: k for k, v in structure._THREE_TO_ONE.items()}


def pdb_bytes(sequence="AC", scores=(69.0, 70.0), chains=("A",), *, af2=True):
    """Minimal fixed-column PDB containing synthetic, nonphysical coordinates."""
    lines = ["HEADER    SOFTWARE TEST FIXTURE"]
    if af2:
        lines.append("TITLE     ALPHAFOLD MONOMER V2.0 SYNTHETIC TEST STRUCTURE")
    atom_id = 0
    for chain in chains:
        for number, (aa, score) in enumerate(zip(sequence, scores), 1):
            for atom_name, element in (("N", "N"), ("CA", "C")):
                atom_id += 1
                lines.append(
                    f"ATOM  {atom_id:5d} {atom_name:^4s} {RESIDUES[aa]:3s} {chain:1s}{number:4d}    "
                    f"{atom_id:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{score:6.2f}          {element:>2s}")
        lines.append("TER")
    lines.append("END")
    return ("\n".join(lines) + "\n").encode("ascii")


def local_structure(tmp_path, *, sequence="AC", scores=(69.0, 70.0), chains=("A",), af2=True):
    source = tmp_path / "inputs"
    source.mkdir()
    path = source / "P49917.pdb"
    path.write_bytes(pdb_bytes(sequence, scores, chains, af2=af2))
    return source, path


def resolved_structure(tmp_path):
    source, path = local_structure(tmp_path)
    resolved = structure.resolve_structures([("P49917", "AC")], tmp_path / "run", source, offline=True)
    return resolved, path


def mock_foldseek(tmp_path, monkeypatch, *, descriptor="P49917.pdb_A\tAC\tpy\n", code=0,
                  version_code=0, write_output=True):
    binary = tmp_path / "foldseek"
    binary.write_bytes(b"synthetic executable identity only\n")
    binary.chmod(0o755)
    calls = []

    def run(args, **kwargs):
        assert isinstance(args, list)
        assert kwargs.get("shell", False) is False
        calls.append((args, kwargs))
        if args[1] == "version":
            return SimpleNamespace(returncode=version_code, stdout="10-test\n", stderr="")
        assert args[1:9] == ["structureto3didescriptor", "-v", "0", "--threads", "1", "--chain-name-mode", "1", str(Path(args[-2]))]
        if write_output:
            Path(args[-1]).write_text(descriptor)
        return SimpleNamespace(returncode=code, stdout="", stderr="software fixture")

    monkeypatch.setattr(structure.subprocess, "run", run)
    return binary, calls


def api_record(**updates):
    row = {"entryId": "AF-P49917-F1", "uniprotAccession": "P49917", "uniprotSequence": "AC",
           "toolUsed": "AlphaFold Monomer v2.0 pipeline",
           "uniprotStart": 1, "uniprotEnd": 2, "latestVersion": 6,
           "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P49917-F1-model_v6.pdb"}
    row.update(updates)
    return row


def test_fasta_normalizes_uniprot_headers_preserves_order_and_sequence(tmp_path):
    path = tmp_path / "input.fasta"
    path.write_text("\ufeff>sp|P49917|LIG4_HUMAN description\na c\nDE\n\n>toy:1|tag hello\nfG\n>tr|Q9H0A0|NAME\nHI\n")
    assert structure.read_fasta(path) == [("P49917", "ACDE"), ("toy:1|tag", "FG"), ("Q9H0A0", "HI")]


@pytest.mark.parametrize("text", ["", "AC\n", ">\nAC", ">A\n", ">A\nAX", ">A\nA*", ">A\nA-",
                                        ">../A\nAC", ">sp||BAD\nAC", ">A\nAC\n>A\nDE",
                                        ">sp|P49917|X\nAC\n>P49917\nDE"])
def test_invalid_fasta_is_rejected(tmp_path, text):
    path = tmp_path / "bad.fasta"
    path.write_text(text)
    with pytest.raises(ValueError):
        structure.read_fasta(path)


def test_complete_local_chain_is_copied_without_changing_input(tmp_path):
    source, original = local_structure(tmp_path)
    before = original.read_bytes()
    resolved = structure.resolve_structures([("P49917", "AC")], tmp_path / "run", source, offline=True)
    assert original.read_bytes() == before
    assert resolved[0]["chain_id"] == "A"
    assert resolved[0]["plddt"] == [69.0, 70.0]
    assert Path(resolved[0]["path"]).read_bytes() == before
    assert resolved[0]["provenance"]["structure_sha256"] == hashlib.sha256(before).hexdigest()
    assert resolved[0]["provenance"]["complete_sequence_match"] is True
    assert json.loads((tmp_path / "run/structures/provenance.json").read_text())[0]["length"] == 2


def test_local_sequence_mismatch_and_ambiguous_chain_are_rejected(tmp_path):
    source, _ = local_structure(tmp_path, chains=("A", "B"))
    with pytest.raises(ValueError, match="Multiple chains"):
        structure.resolve_structures([("P49917", "AC")], tmp_path / "ambiguous", source, offline=True)
    with pytest.raises(ValueError, match="No complete structure chain"):
        structure.resolve_structures([("P49917", "AD")], tmp_path / "mismatch", source, offline=True)


def test_experimental_b_factors_are_not_assumed_to_be_plddt(tmp_path):
    source, path = local_structure(tmp_path, af2=False)
    with pytest.raises(ValueError, match="experimental B-factors are not pLDDT"):
        structure.resolve_structures([("P49917", "AC")], tmp_path / "rejected", source, offline=True)
    Path(str(path) + ".json").write_text(json.dumps({"structure_source": "alphafold2", "bfactor_is_plddt": True}))
    resolved = structure.resolve_structures([("P49917", "AC")], tmp_path / "declared", source, offline=True)
    assert resolved[0]["provenance"]["confidence_source"] == "user-declared AF2 sidecar"


def test_multiple_local_formats_are_not_arbitrarily_selected(tmp_path):
    source, _ = local_structure(tmp_path)
    (source / "P49917.cif").write_text("software fixture")
    with pytest.raises(ValueError, match="multiple local structure"):
        structure.resolve_structures([("P49917", "AC")], tmp_path / "run", source, offline=True)


def test_offline_missing_structure_never_attempts_download(tmp_path, monkeypatch):
    source = tmp_path / "inputs"
    source.mkdir()
    monkeypatch.setattr(structure, "_fetch_bytes", lambda *a, **k: pytest.fail("unexpected network request"))
    with pytest.raises(ValueError, match="offline mode forbids"):
        structure.resolve_structures([("P49917", "AC")], tmp_path / "run", source, offline=True)


def test_afdb_selection_is_exact_and_not_first_record():
    full = api_record()
    assert structure._select_afdb_record([api_record(uniprotEnd=1), full], "P49917", "AC") == full
    for rows in ([full, full], [api_record(uniprotAccession="P49917-2")], [api_record(uniprotSequence="AD")],
                 [api_record(uniprotStart=2)], [api_record(latestVersion=None)],
                 [api_record(toolUsed="AlphaFold 3")], [api_record(isComplex=True)],
                 [api_record(sequence="AD")], [api_record(entryId="AF-P49917-F2")], []):
        with pytest.raises(ValueError):
            structure._select_afdb_record(rows, "P49917", "AC")


def test_mocked_afdb_download_validates_metadata_and_actual_chain(tmp_path, monkeypatch):
    row = api_record()
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        if kwargs.get("api"):
            return json.dumps([row]).encode()
        return pdb_bytes()

    monkeypatch.setattr(structure, "_fetch_bytes", fetch)
    result = structure.resolve_structures([("P49917", "AC")], tmp_path / "run")
    assert len(calls) == 2
    assert result[0]["provenance"]["version"] == 6
    assert result[0]["provenance"]["source"] == "alphafold_database"
    assert (tmp_path / "run/structures/P49917.afdb.json").is_file()


@pytest.mark.parametrize("updates, structure_bytes, message", [
    ({"pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P49917-F1-model_v4.pdb"}, pdb_bytes(), "entry/version"),
    ({}, pdb_bytes("AD"), "No complete structure chain"),
])
def test_afdb_version_or_actual_sequence_mismatch_is_rejected(tmp_path, monkeypatch, updates, structure_bytes, message):
    def fetch(url, **kwargs):
        return json.dumps([api_record(**updates)]).encode() if kwargs.get("api") else structure_bytes

    monkeypatch.setattr(structure, "_fetch_bytes", fetch)
    with pytest.raises(ValueError, match=message):
        structure.resolve_structures([("P49917", "AC")], tmp_path / "run")
    assert not (tmp_path / "run/structures/provenance.json").exists()


def test_af3_local_structure_is_not_accepted_as_af2(tmp_path):
    source, path = local_structure(tmp_path)
    path.write_bytes(path.read_bytes().replace(b"ALPHAFOLD MONOMER V2.0", b"ALPHAFOLD 3 PREDICTION"))
    with pytest.raises(ValueError, match="not identified as AF2"):
        structure.resolve_structures([("P49917", "AC")], tmp_path / "run", source, offline=True)


@pytest.mark.parametrize("url", ["http://alphafold.ebi.ac.uk/files/A.pdb", "https://example.com/files/A.pdb",
                                     "https://user:secret@alphafold.ebi.ac.uk/files/A.pdb",
                                     "https://alphafold.ebi.ac.uk/other/A.pdb"])
def test_only_official_https_structure_urls_are_accepted(url):
    with pytest.raises(ValueError, match="official HTTPS"):
        structure._checked_url(url)


def test_fresh_subdirectories_prevent_overwrite_and_symlink_outputs(tmp_path):
    structure._new_subdir(tmp_path / "run", "structures")
    with pytest.raises(FileExistsError):
        structure._new_subdir(tmp_path / "run", "structures")
    (tmp_path / "run/tokens").symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(FileExistsError):
        structure._new_subdir(tmp_path / "run", "tokens")


def test_foldseek_tokens_use_exact_chain_and_mean_atom_plddt_mask(tmp_path, monkeypatch):
    resolved, source = resolved_structure(tmp_path)
    original = source.read_bytes()
    binary, calls = mock_foldseek(tmp_path, monkeypatch)
    records, provenance = structure.make_tokens(resolved, binary, tmp_path / "run")
    assert records == [("P49917", "A#Cy")]
    assert provenance["records"][0]["masked_residues"] == 1
    assert provenance["foldseek_version"] == "10-test"
    assert provenance["foldseek_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert source.read_bytes() == original
    assert len(calls) == 2
    assert not list((tmp_path / "run/tokens").glob("foldseek-*"))
    assert (tmp_path / "run/tokens/tokens.fasta").read_text() == ">P49917\nA#Cy\n"


@pytest.mark.parametrize("descriptor", ["P49917.pdb_B\tAC\tpy\n", "P49917.pdb_A\tAD\tpy\n",
                                            "P49917.pdb_A\tAC\tp\n", "P49917.pdb_A\tAC\tpx\n",
                                            "P49917.pdb\tAC\tpy\n", "malformed\n",
                                            "P49917.pdb_A\tAC\tpy\nP49917.pdb_A\tAC\tpy\n"])
def test_inexact_or_ambiguous_foldseek_descriptors_are_rejected(tmp_path, monkeypatch, descriptor):
    resolved, _ = resolved_structure(tmp_path)
    binary, _ = mock_foldseek(tmp_path, monkeypatch, descriptor=descriptor)
    with pytest.raises(ValueError):
        structure.make_tokens(resolved, binary, tmp_path / "run")
    assert not (tmp_path / "run/tokens/tokens.fasta").exists()


@pytest.mark.parametrize("code,write_output,version_code", [(1, True, 0), (0, False, 0), (0, True, 1)])
def test_failed_foldseek_never_yields_tokens(tmp_path, monkeypatch, code, write_output, version_code):
    resolved, _ = resolved_structure(tmp_path)
    binary, _ = mock_foldseek(tmp_path, monkeypatch, code=code, write_output=write_output, version_code=version_code)
    with pytest.raises(ValueError, match="failed"):
        structure.make_tokens(resolved, binary, tmp_path / "run")
    assert not (tmp_path / "run/tokens/tokens.fasta").exists()


def test_changed_structure_is_rejected_before_foldseek_descriptor(tmp_path, monkeypatch):
    resolved, _ = resolved_structure(tmp_path)
    path = Path(resolved[0]["path"])
    path.write_bytes(path.read_bytes() + b"REMARK changed\n")
    binary, calls = mock_foldseek(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="changed after"):
        structure.make_tokens(resolved, binary, tmp_path / "run")
    assert len(calls) == 1


@pytest.mark.parametrize("confidence", [[], [70.0], [70.0, float("nan")], [70.0, 101.0]])
def test_missing_or_invalid_confidence_is_hard_failure(tmp_path, monkeypatch, confidence):
    resolved, _ = resolved_structure(tmp_path)
    resolved[0]["plddt"] = confidence
    binary, calls = mock_foldseek(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="pLDDT coverage"):
        structure.make_tokens(resolved, binary, tmp_path / "run")
    assert len(calls) == 1


def test_foldseek_timeout_is_clear_failure(tmp_path, monkeypatch):
    resolved, _ = resolved_structure(tmp_path)
    binary, _ = mock_foldseek(tmp_path, monkeypatch)

    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 30)

    monkeypatch.setattr(structure.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="Cannot execute Foldseek"):
        structure.make_tokens(resolved, binary, tmp_path / "run")


def test_residue_confidence_is_mean_of_all_atomic_b_factors(tmp_path):
    source, path = local_structure(tmp_path)
    lines = path.read_text().splitlines()
    atom_rows = [i for i, line in enumerate(lines) if line.startswith("ATOM")]
    # First residue has N=68 and CA=72, so its confidence is exactly 70.
    for i, value in zip(atom_rows[:2], [68.0, 72.0]):
        lines[i] = lines[i][:60] + f"{value:6.2f}" + lines[i][66:]
    path.write_text("\n".join(lines) + "\n")
    resolved = structure.resolve_structures([("P49917", "AC")], tmp_path / "run", source, offline=True)
    assert resolved[0]["plddt"] == [70.0, 70.0]


def test_nonstandard_residue_and_incomplete_chains_are_rejected(tmp_path):
    source, path = local_structure(tmp_path)
    original = path.read_bytes()
    path.write_bytes(original.replace(b"CYS", b"UNK"))
    with pytest.raises(ValueError, match="Nonstandard protein residue"):
        structure.resolve_structures([("P49917", "AC")], tmp_path / "unknown", source, offline=True)
    path.write_bytes(original)
    with pytest.raises(ValueError, match="No complete structure chain"):
        structure.resolve_structures([("P49917", "ACD")], tmp_path / "incomplete", source, offline=True)
