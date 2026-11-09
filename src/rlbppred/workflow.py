"""Traceable FASTA inference with the fixed pretrained representation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


def predict_fasta(fasta, output_dir, *, structure_dir=None, structure_source='alphafold',
                  offline=False, saprot_model=None, foldseek=None, model_dir=None,
                  assets_dir=None, device='cpu', batch_size=256, threshold=.98,
                  max_residues=None):
    from . import __version__, encoder, io, runtime, structure
    from .cli import default_assets_dir, default_model_dir

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch-size must be positive')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite and in [0, 1]')
    if max_residues is not None and (isinstance(max_residues, bool) or not isinstance(max_residues, int) or max_residues < 1):
        raise ValueError('max-residues must be positive')
    destination = Path(output_dir).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Run directory already exists: {destination}')
    records = structure.read_fasta(fasta)
    if max_residues is not None and any(len(seq) > max_residues for _, seq in records):
        raise ValueError('A FASTA sequence exceeds max-residues; no sequences are truncated')
    cache = Path(assets_dir) if assets_dir is not None else default_assets_dir()
    checkpoint = Path(saprot_model) if saprot_model is not None else cache / 'SaProt_650M_AF2.pt'
    binary = Path(foldseek) if foldseek is not None else cache / 'foldseek/bin/foldseek'
    bundle = Path(model_dir) if model_dir is not None else default_model_dir()
    for name, path in [('SaProt checkpoint', checkpoint), ('Foldseek', binary),
                       ('MLP manifest', bundle / 'model.json'), ('MLP weights', bundle / 'model.safetensors')]:
        if not path.is_file():
            raise ValueError(f'{name} not found: {path}. Run rlbppred setup first or supply its path.')
    destination.mkdir(parents=True)
    receipt = {'rlbppred_version': __version__, 'started_utc': datetime.now(timezone.utc).isoformat(),
               'fasta_sha256': hashlib.sha256(Path(fasta).read_bytes()).hexdigest(),
               'proteins': len(records), 'device': device, 'screening_threshold': threshold,
               'operator': '>=', 'status': 'running'}
    receipt_path = destination / 'run.json'
    def save_receipt():
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    save_receipt()
    try:
        print('Matching complete AF2 structures...', file=sys.stderr)
        resolved = structure.resolve_structures(records, destination, structure_dir=structure_dir,
                                                source=structure_source, offline=offline)
        print('Generating AA+3Di tokens (pLDDT < 70 masked)...', file=sys.stderr)
        tokens, token_metadata = structure.make_tokens(resolved, binary, destination, plddt_threshold=70.)
        print('Extracting fixed SaProt-650M-AF2 global-mean features...', file=sys.stderr)
        ids, matrix, feature_metadata = encoder.encode_tokens(tokens, checkpoint, device=device,
                                                               output=destination / 'features.npz',
                                                               max_residues=max_residues)
        print('Scoring with the frozen 300-member strict MLP...', file=sys.stderr)
        model = runtime.load_model(bundle, device=device)
        scores = runtime.predict_embeddings(model, matrix, batch_size=batch_size)
        io.write_predictions(destination / 'predictions.csv', ids, scores, threshold)
        receipt.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                       model_id=runtime.MODEL_ID, foldseek_version=token_metadata['foldseek_version'],
                       encoder_sha256=feature_metadata['checkpoint_sha256'],
                       above_threshold=int((scores >= threshold).sum()),
                       note='High scores are not automatically novel candidates; deployment FPR/FDR is not guaranteed.')
        save_receipt()
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save_receipt()
        raise
    return receipt
