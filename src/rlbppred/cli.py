"""Frozen inference CLI. Explicit asset setup; no fitting or silent overwrites."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from . import __version__


def default_model_dir():
    return Path(os.environ.get('RLBPPRED_MODEL_DIR', Path.home() / '.cache/rlbppred/strict-saprot-af2-mlp-v1')).expanduser()


def default_assets_dir():
    return Path(os.environ.get('RLBPPRED_ASSETS_DIR', Path.home() / '.cache/rlbppred/assets')).expanduser()


def parser():
    p = argparse.ArgumentParser(description='Score raw SaProt-AF2 embeddings with the frozen strict RLBPPred MLP ensemble.')
    p.add_argument('--version', action='version', version=f'rlbppred {__version__}')
    subs = p.add_subparsers(dest='command', required=True)
    setup = subs.add_parser('setup', help='Explicitly download and verify fixed pretrained assets (about 3.6 GB)')
    setup.add_argument('--components', choices=['all', 'model', 'encoder', 'foldseek'], default='all')
    setup.add_argument('--assets-dir', type=Path, default=default_assets_dir())
    setup.add_argument('--model-dir', type=Path, default=default_model_dir())
    fasta = subs.add_parser('predict-fasta', help='Matched AF2 structures -> masked 3Di -> SaProt -> frozen MLP scores')
    fasta.add_argument('--fasta', required=True, type=Path)
    fasta.add_argument('--output-dir', required=True, type=Path, help='New directory; stores provenance, features and predictions.csv')
    fasta.add_argument('--structure-dir', type=Path, help='Optional local ID.pdb/ID.cif AF2 structures')
    fasta.add_argument('--structure-source', choices=['alphafold', 'local'], default='alphafold')
    fasta.add_argument('--offline', action='store_true', help='Forbid structure downloads; requires local structures')
    fasta.add_argument('--assets-dir', type=Path, default=default_assets_dir())
    fasta.add_argument('--saprot-model', type=Path, help='Override the fixed official encoder checkpoint location')
    fasta.add_argument('--foldseek', type=Path, help='Override the Foldseek executable location; version is recorded')
    fasta.add_argument('--model-dir', type=Path, default=default_model_dir())
    fasta.add_argument('--device', default='cpu')
    fasta.add_argument('--batch-size', type=int, default=256)
    fasta.add_argument('--threshold', type=float, default=.98)
    fasta.add_argument('--max-residues', type=int, help='Explicit resource limit; exceeding inputs fail, never truncate')
    predict = subs.add_parser('predict', help='Predict scores from a safe NPZ feature matrix')
    predict.add_argument('--embeddings', required=True, type=Path, help='NPZ with protein_id and raw float32 X (N x 1280)')
    predict.add_argument('--output', required=True, type=Path)
    predict.add_argument('--model-dir', type=Path, default=default_model_dir())
    predict.add_argument('--device', default='cpu')
    predict.add_argument('--batch-size', type=int, default=256)
    predict.add_argument('--threshold', type=float, default=.98, help='Screening flag only, not a calibrated probability/FDR guarantee')
    pack = subs.add_parser('pack', help='Pack matching legacy per-protein tensors into a safe NPZ')
    pack.add_argument('--embedding-dir', type=Path, required=True)
    pack.add_argument('--ids', type=Path, required=True, help='Whitespace-delimited protein IDs in required output order')
    pack.add_argument('--output', type=Path, required=True)
    model = subs.add_parser('model', help='Manage an offline, trusted pretrained bundle')
    ms = model.add_subparsers(dest='model_command', required=True)
    export = ms.add_parser('export', help='Convert only the hash-pinned original checkpoint into safetensors')
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--output-dir', type=Path, required=True)
    install = ms.add_parser('install', help='Validate and copy a supplied bundle into the model cache')
    install.add_argument('--source', type=Path, required=True)
    install.add_argument('--model-dir', type=Path, default=default_model_dir())
    verify = ms.add_parser('verify', help='Check hashes, architecture, scaler and all 300 model states')
    verify.add_argument('--model-dir', type=Path, default=default_model_dir())
    subs.add_parser('doctor', help='Report dependencies and local model availability without downloading anything')
    return p


def _run(args):
    if args.command == 'setup':
        from . import assets
        installed = {}
        if args.components in ('all', 'model'):
            installed['model'] = str(assets.download_model(args.model_dir, cache_dir=args.assets_dir))
        if args.components in ('all', 'encoder'):
            installed['encoder'] = str(assets.download_saprot(args.assets_dir))
        if args.components in ('all', 'foldseek'):
            installed['foldseek'] = str(assets.download_foldseek(args.assets_dir))
        print(json.dumps(installed, indent=2))
        return 0
    if args.command == 'predict-fasta':
        from .workflow import predict_fasta
        result = predict_fasta(args.fasta, args.output_dir,
                              structure_dir=args.structure_dir, structure_source=args.structure_source,
                              offline=args.offline, saprot_model=args.saprot_model,
                              foldseek=args.foldseek, model_dir=args.model_dir,
                              assets_dir=args.assets_dir, device=args.device,
                              batch_size=args.batch_size, threshold=args.threshold,
                              max_residues=args.max_residues)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == 'doctor':
        versions = {}
        for name in ['numpy', 'torch', 'safetensors']:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        result = {'rlbppred': __version__, 'python': sys.version.split()[0], 'dependencies': versions,
                  'model_dir': str(default_model_dir()), 'model_present': (default_model_dir() / 'model.json').is_file()}
        print(json.dumps(result, indent=2))
        return 0 if all(versions.values()) else 1
    from . import io, runtime
    if args.command == 'pack':
        print(io.pack_embedding_directory(args.embedding_dir, args.ids, args.output))
        return 0
    if args.command == 'model':
        if args.model_command == 'export':
            print(runtime.export_model(args.checkpoint, args.output_dir))
        elif args.model_command == 'verify':
            model = runtime.load_model(args.model_dir)
            print(json.dumps(model.metadata, indent=2))
        else:
            if args.model_dir.exists():
                raise FileExistsError(f'Model destination already exists: {args.model_dir}')
            model = runtime.load_model(args.source)
            del model
            args.model_dir.parent.mkdir(parents=True, exist_ok=True)
            # Only these two validated files are installed; no scripts or pickles.
            args.model_dir.mkdir()
            try:
                for name in ['model.json', 'model.safetensors']:
                    shutil.copyfile(args.source / name, args.model_dir / name)
                runtime.load_model(args.model_dir)
            except Exception:
                shutil.rmtree(args.model_dir)
                raise
            print(args.model_dir)
        return 0
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    if not 0 <= args.threshold <= 1:
        raise ValueError('threshold must be in [0, 1]')
    ids, X = io.load_npz(args.embeddings)
    if not (args.model_dir / 'model.json').is_file():
        raise ValueError('Model bundle not installed. Use rlbppred model install --source <trusted-bundle>, or --model-dir.')
    model = runtime.load_model(args.model_dir, device=args.device)
    probabilities = runtime.predict_embeddings(model, X, batch_size=args.batch_size)
    io.write_predictions(args.output, ids, probabilities, args.threshold)
    print(json.dumps({'output': str(args.output), 'proteins': len(ids), 'model_id': runtime.MODEL_ID,
                      'screening_threshold': args.threshold, 'operator': '>=',
                      'above_threshold': int((probabilities >= args.threshold).sum()),
                      'note': 'High scores alone do not establish novel candidate status or deployment FPR.'}))
    return 0


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        return _run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'rlbppred: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
