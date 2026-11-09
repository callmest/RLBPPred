"""Validated, non-pickle feature inputs and exclusive prediction outputs."""
import csv
import pickle
from pathlib import Path
import re

import numpy as np


def validate_ids(values):
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0 or array.dtype.kind not in 'US':
        raise ValueError('protein_id must be a nonempty 1-D string array, not objects/pickle')
    ids = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in array]
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:|+\-]*', x) for x in ids):
        raise ValueError('Protein IDs must be nonempty ASCII identifiers without whitespace or path separators')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate protein IDs are not allowed')
    return ids


def load_npz(path):
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            if not {'protein_id', 'X'}.issubset(data.files):
                raise ValueError('NPZ requires protein_id and X arrays')
            ids = validate_ids(data['protein_id'])
            X = np.array(data['X'], copy=True)
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError(f'Cannot read embedding NPZ: {exc}') from exc
    if X.dtype != np.dtype('float32'):
        raise ValueError('X must have float32 dtype; supply raw, unstandardized SaProt-AF2 embeddings')
    if X.shape != (len(ids), 1280):
        raise ValueError('X must have shape N x 1280 matching protein_id')
    if not np.isfinite(X).all():
        raise ValueError('X contains non-finite values (NaN or Inf)')
    return ids, X


def write_predictions(output, ids, probabilities, threshold=.98):
    ids = validate_ids(np.asarray(ids))
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(ids),) or not np.isfinite(probabilities).all():
        raise ValueError('Expected one finite score per protein')
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError('Scores must be in [0, 1]')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be finite and in [0, 1]')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['protein_id', 'probability', 'classification_at_0_5', 'above_screening_threshold'])
        for protein, score in zip(ids, probabilities):
            writer.writerow([protein, repr(float(score)), int(score >= .5), int(score >= threshold)])
    return output


def pack_embedding_directory(directory, ids_path, output):
    """Pack historical tensor dictionaries using PyTorch's restricted loader only."""
    import torch
    ids = validate_ids(np.asarray(Path(ids_path).read_text().split()))
    requested_output = Path(output).expanduser()
    if requested_output.is_symlink():
        raise FileExistsError(f'Output is a symbolic link: {requested_output}')
    directory, output = Path(directory).resolve(), requested_output.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    vectors = []
    for protein in ids:
        path = directory / (protein + '.pt')
        try:
            item = torch.load(path, map_location='cpu', weights_only=True)
        except (pickle.UnpicklingError, EOFError, RuntimeError, OSError, ValueError) as exc:
            raise ValueError(f'{protein}: cannot read a safe embedding tensor; arbitrary pickle is not accepted') from exc
        if not isinstance(item, dict) or not isinstance(item.get('embeddings'), torch.Tensor):
            raise ValueError(f'{protein}: expected a tensor dictionary with embeddings')
        if item.get('pooling') != 'mean' or str(item.get('model', '')).lower() != 'saprot-650m-af2':
            raise ValueError(f'{protein}: expected mean-pooled saprot-650m-af2 metadata')
        vector = item['embeddings'].detach().cpu().numpy()
        if vector.shape != (1280,) or vector.dtype != np.dtype('float32') or not np.isfinite(vector).all():
            raise ValueError(f'{protein}: invalid float32 1280-vector')
        vectors.append(vector)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        np.savez_compressed(handle, protein_id=np.asarray(ids), X=np.stack(vectors))
    return output
