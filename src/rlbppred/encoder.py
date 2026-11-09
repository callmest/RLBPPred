"""Frozen SaProt-650M-AF2 embeddings using the original fair-esm backend.

Inputs are interleaved amino-acid/3Di tokens, not amino-acid-only sequences.
The fixed upstream checkpoint is authenticated before restricted loading. No
training, sequence truncation, chunking, or alternative encoder is performed.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import tempfile


CHECKPOINT_SHA256 = "bc110567bc75511c7f46eb3e7d109fd6e81051baad471e22c5306ed71653cd52"
CHECKPOINT_BYTES = 2_606_464_143
CHECKPOINT_REVISION = "1b8cddd7bad34f6f5627422b4908309840c78fef"
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY#"
STRUCTURE_VOCAB = "pynwrqhgdlvtmfsaeikc#"
_SPECIAL_TOKENS = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
_CONFIG = {"num_layers": 33, "embed_dim": 1280, "attention_heads": 20,
           "token_dropout": True}


def _esm_backend():
    """Reject the unrelated ``esm`` distribution and shadowing local modules."""
    try:
        distribution = importlib.metadata.distribution("fair-esm")
        if distribution.version != "2.0.0":
            raise RuntimeError("SaProt encoding requires exactly fair-esm==2.0.0")
        esm = importlib.import_module("esm")
        data = importlib.import_module("esm.data")
        esm2 = importlib.import_module("esm.model.esm2")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            "Install fair-esm==2.0.0 for SaProt encoding; the ESM-C package named "
            "'esm' and transformers backends are not interchangeable"
        ) from exc
    for module, relative in ((esm, "esm/__init__.py"), (data, "esm/data.py"),
                             (esm2, "esm/model/esm2.py")):
        actual = getattr(module, "__file__", None)
        expected = Path(distribution.locate_file(relative)).resolve()
        if actual is None or Path(actual).resolve() != expected:
            raise RuntimeError(
                "Imported esm modules do not belong to fair-esm==2.0.0; "
                "remove the conflicting ESM-C installation or local esm module"
            )
    if not callable(getattr(data, "Alphabet", None)) or not callable(getattr(esm2, "ESM2", None)):
        raise RuntimeError("fair-esm is missing its Alphabet or ESM2 implementation")
    return data.Alphabet, esm2.ESM2


def _alphabet(alphabet_class):
    tokens = _SPECIAL_TOKENS + [aa + structure for aa, structure in
                                itertools.product(AA_VOCAB, STRUCTURE_VOCAB)]
    alphabet = alphabet_class(standard_toks=tokens, prepend_toks=[], append_toks=[],
                              prepend_bos=True, append_eos=True, use_msa=False)
    # fair-esm pads the 446-token list to 448. This exact two-token removal is
    # part of the upstream SaProt alphabet, not a generic ESM tokenizer.
    alphabet.all_toks = alphabet.all_toks[:-2]
    alphabet.unique_no_split_tokens = alphabet.all_toks
    alphabet.tok_to_idx = {token: index for index, token in enumerate(alphabet.all_toks)}
    if alphabet.all_toks != tokens or len(alphabet.tok_to_idx) != 446:
        raise RuntimeError("Unexpected fair-esm alphabet layout; refusing incompatible tokenization")
    if (alphabet.cls_idx, alphabet.padding_idx, alphabet.eos_idx, alphabet.unk_idx,
            alphabet.mask_idx) != (0, 1, 2, 3, 4):
        raise RuntimeError("Unexpected SaProt special-token indices")
    return alphabet


def _validated_records(records, max_residues):
    import numpy as np
    from .io import validate_ids

    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("records must be a nonempty sequence of (protein_id, AA+3Di tokens)")
    if max_residues is not None and (isinstance(max_residues, bool)
                                     or not isinstance(max_residues, int)
                                     or max_residues < 1):
        raise ValueError("max_residues must be a positive integer or None")
    if any(not isinstance(record, (list, tuple)) or len(record) != 2 for record in records):
        raise ValueError("Each record must contain exactly protein_id and AA+3Di tokens")
    if any(not isinstance(record[0], str) for record in records):
        raise ValueError("Protein IDs must be strings")
    ids = validate_ids(np.asarray([record[0] for record in records]))
    sequences = []
    for protein, record in zip(ids, records):
        combined = record[1]
        if not isinstance(combined, str) or not combined or len(combined) % 2:
            raise ValueError(f"{protein}: expected a nonempty even-length AA+3Di string")
        pairs = [combined[index:index + 2] for index in range(0, len(combined), 2)]
        for position, pair in enumerate(pairs, 1):
            if pair[0] not in AA_VOCAB or pair[1] not in STRUCTURE_VOCAB:
                raise ValueError(f"{protein}: invalid AA+3Di pair at residue {position}: {pair!r}")
        if max_residues is not None and len(pairs) > max_residues:
            raise ValueError(
                f"{protein}: {len(pairs)} residues exceeds explicit max_residues={max_residues}; "
                "sequences are not truncated or chunked"
            )
        sequences.append(pairs)
    return ids, sequences


def _load_checkpoint(checkpoint, torch):
    source = Path(checkpoint).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("SaProt checkpoint must be a regular file")
    with source.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if before.st_size != CHECKPOINT_BYTES:
            raise ValueError("SaProt checkpoint byte size mismatch; refusing deserialization")
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("SaProt checkpoint SHA-256 mismatch; refusing deserialization")
        handle.seek(0)
        try:
            checkpoint_data = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                "The authenticated SaProt checkpoint could not be read with the restricted "
                "weights_only loader; arbitrary pickle fallback is not permitted"
            ) from exc
        after = os.fstat(handle.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("SaProt checkpoint changed during loading")
    if not isinstance(checkpoint_data, Mapping):
        raise ValueError("SaProt checkpoint must contain model and config mappings")
    config, weights = checkpoint_data.get("config"), checkpoint_data.get("model")
    if not isinstance(config, Mapping) or not isinstance(weights, Mapping) or not weights:
        raise ValueError("SaProt checkpoint is missing its model weights or config")
    for name, expected in _CONFIG.items():
        if config.get(name) != expected:
            raise ValueError(f"SaProt checkpoint architecture mismatch: {name}")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
           for key, value in weights.items()):
        raise ValueError("SaProt model weights must be a string-to-tensor mapping")
    return weights


def _output_path(output):
    if output is None:
        return None
    requested = Path(output).expanduser()
    if requested.is_symlink() or requested.exists():
        raise FileExistsError(f"Output already exists or is a symbolic link: {requested}")
    return requested.resolve()


def _write_npz(output, ids, matrix, metadata):
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=".embedding-", suffix=".npz",
                                          dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, protein_id=np.asarray(ids), X=matrix,
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
            handle.flush()
            os.fsync(handle.fileno())
        # Publish a complete file atomically and exclusively; never replace an
        # existing output, including one created concurrently during inference.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def encode_tokens(records, checkpoint, device="cpu", output=None, max_residues=None):
    """Return IDs, raw float32 N×1280 embeddings, and provenance metadata.

    ``records`` contains ``(id, combined_tokens)`` pairs such as ``("P1", "ApCy")``.
    Each amino acid is immediately followed by its 3Di state. The final-layer
    residue vectors are globally mean pooled after excluding BOS and EOS. An
    optional residue limit fails explicitly; no implicit 1,024-residue limit is
    imposed. Long inputs can require substantial device memory. Device errors
    are not redirected to another device. If requested, ``output`` is a new
    non-pickle NPZ containing ``protein_id``, ``X``, and JSON-string ``metadata``.
    """
    import numpy as np
    from .runtime import _device, _torch

    ids, sequences = _validated_records(records, max_residues)
    destination = _output_path(output)
    torch = _torch()
    selected_device = _device(torch, device)
    alphabet_class, model_class = _esm_backend()
    alphabet = _alphabet(alphabet_class)
    weights = _load_checkpoint(checkpoint, torch)
    model = model_class(alphabet=alphabet, **_CONFIG)
    missing = set(model.state_dict()) - set(weights)
    unexpected = set(weights) - set(model.state_dict())
    omitted_contact_head = missing == {
        "contact_head.regression.weight", "contact_head.regression.bias"
    } and not unexpected
    if omitted_contact_head:
        # The authenticated upstream checkpoint omits only the two contact-
        # prediction regression parameters. That head is never called when
        # extracting repr_layers=[33]. Remove it explicitly so every remaining
        # encoder/LM parameter is loaded strictly, with no random head retained.
        model.contact_head = None
    # Never merge into a freshly initialized state dict: missing or unexpected
    # keys must fail instead of silently retaining random parameters.
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise ValueError("SaProt checkpoint does not strictly match the ESM2 model; no keys were ignored") from exc
    del weights
    model = model.to(device=selected_device, dtype=torch.float32).eval()
    if model.num_layers != 33:
        raise ValueError("Expected the final (33rd) layer of SaProt-650M-AF2")
    matrix = np.empty((len(ids), 1280), dtype=np.float32)
    with torch.no_grad():
        for row, (protein, pairs) in enumerate(zip(ids, sequences)):
            token_ids = [alphabet.cls_idx] + [alphabet.tok_to_idx[pair] for pair in pairs] + [alphabet.eos_idx]
            if alphabet.unk_idx in token_ids:
                raise ValueError(f"{protein}: unknown tokens are not accepted")
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=selected_device)
            results = model(input_ids, repr_layers=[33])
            hidden = results["representations"][33]
            if tuple(hidden.shape) != (1, len(pairs) + 2, 1280) or hidden.dtype != torch.float32:
                raise ValueError(f"{protein}: unexpected SaProt residue-representation shape or dtype")
            vector = hidden[0, 1:-1, :].mean(dim=0).detach().cpu().numpy()
            if not np.isfinite(vector).all():
                raise ValueError(f"{protein}: SaProt produced non-finite embeddings")
            matrix[row] = vector
            del results, hidden, vector, input_ids
    metadata = {
        "schema_version": 1,
        "model": "saprot-650m-af2",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoint_repository": "westlake-repl/SaProt_650M_AF2",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "backend": "fair-esm",
        "backend_version": "2.0.0",
        "omitted_contact_prediction_head": omitted_contact_head,
        "architecture": dict(_CONFIG),
        "representation_layer": 33,
        "embedding_dim": 1280,
        "dtype": "float32",
        "pooling": "global-mean-excluding-special-tokens",
        "alphabet_size": 446,
        "amino_acid_vocab": AA_VOCAB,
        "structure_vocab": STRUCTURE_VOCAB,
        "residue_counts": [len(pairs) for pairs in sequences],
        "max_residues": max_residues,
        "truncated": False,
        "chunked": False,
        "device": selected_device,
    }
    if destination is not None:
        _write_npz(destination, ids, matrix, metadata)
    return ids, matrix, metadata
