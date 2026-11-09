"""Strict FASTA, AlphaFold structure and Foldseek 3Di preparation.

No structure is predicted here. AlphaFold DB retrieval is accession-based;
unmatched/new sequences need an explicitly supplied, sequence-matched AF2
structure. Inputs are never silently truncated, replaced or skipped.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from . import __version__


AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWY')
STRUCTURE_ALPHABET = frozenset('pynwrqhgdlvtmfsaeikc#')
AFDB_API = 'https://alphafold.ebi.ac.uk/api/prediction/'
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:|+\-]*\Z')
_UNIPROT = re.compile(
    r'(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[1-9][0-9]*)?\Z')
_AF2 = re.compile(r'ALPHAFOLD(?:[\s_-]*2\b|[\s_-]+(?:MONOMER|MULTIMER)[\s_-]+V?2(?:\.|\b))')
_THREE_TO_ONE = dict(zip(
    'ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split(),
    'ACDEFGHIKLMNPQRSTVWY'))


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _sequence_digest(sequence):
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def _validate_records(records):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError('FASTA must contain at least one protein')
    seen = set()
    for protein, sequence in records:
        if not isinstance(protein, str) or not _ID.fullmatch(protein):
            raise ValueError('Protein identifiers must be safe nonempty ASCII identifiers')
        if protein in seen:
            raise ValueError(f'Duplicate FASTA identifier: {protein}')
        seen.add(protein)
        if not isinstance(sequence, str) or not sequence or set(sequence) - AMINO_ACIDS:
            raise ValueError(f'{protein}: expected a nonempty sequence containing the 20 standard amino acids')


def read_fasta(path):
    """Read standard protein FASTA; sp|ACCESSION|NAME headers become ACCESSION.

    Simple first-token identifiers are preserved. Sequence letters are
    uppercased; whitespace is ignored, but X, stop characters and gaps are not.
    """
    records = []
    protein = None
    pieces = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith('>'):
                if protein is not None:
                    records.append((protein, ''.join(pieces).upper()))
                header = line[1:].split()
                if not header:
                    raise ValueError(f'FASTA line {number}: empty header')
                protein = header[0]
                fields = protein.split('|')
                if fields[0] in ('sp', 'tr'):
                    if len(fields) != 3 or not fields[1]:
                        raise ValueError(f'FASTA line {number}: malformed UniProt header')
                    protein = fields[1]
                pieces = []
            else:
                if protein is None:
                    raise ValueError(f'FASTA line {number}: sequence appears before a header')
                pieces.append(''.join(line.split()))
    if protein is not None:
        records.append((protein, ''.join(pieces).upper()))
    _validate_records(records)
    return records


def _new_subdir(workdir, name):
    workdir = Path(workdir).expanduser()
    if workdir.is_symlink():
        raise ValueError('Run directory must not be a symbolic link')
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Output directory already exists: {target}')
    target.mkdir()
    return target


def _checked_url(url, api=False):
    parsed = urlparse(url)
    prefix = '/api/prediction/' if api else '/files/'
    if (parsed.scheme != 'https' or parsed.hostname != 'alphafold.ebi.ac.uk'
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not parsed.path.startswith(prefix)):
        raise ValueError('Expected an official HTTPS AlphaFold DB URL')
    return url


def _fetch_bytes(url, *, api=False, max_bytes=100_000_000):
    """Bounded public download; no user sequences or credentials are submitted."""
    _checked_url(url, api=api)
    request = Request(url, headers={'User-Agent': f'RLBPPred/{__version__} structure retrieval',
                                    'Accept': 'application/json' if api else '*/*'})
    try:
        with urlopen(request, timeout=45) as response:
            _checked_url(response.geturl(), api=api)
            content = response.read(max_bytes + 1)
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        raise ValueError(f'AlphaFold DB retrieval failed: {exc}') from exc
    if len(content) > max_bytes:
        raise ValueError('AlphaFold DB response exceeds the safety size limit')
    return content


def _af2_local_identity(path):
    # AFDB exports identify their prediction source in PDB headers or mmCIF
    # metadata. A deliberately supplied sidecar can declare a custom AF2 export.
    with path.open('rb') as stream:
        header = stream.read(131072).decode('utf-8', errors='replace').upper()
    if _AF2.search(header):
        return {'confidence_source': 'AF2 structure metadata', 'bfactor_is_plddt': True}
    sidecar = Path(str(path) + '.json')
    if sidecar.is_file() and not sidecar.is_symlink():
        declaration = json.loads(sidecar.read_text())
        if (declaration.get('structure_source') in ('alphafold2', 'af2')
                and declaration.get('bfactor_is_plddt') is True):
            return {'confidence_source': 'user-declared AF2 sidecar',
                    'bfactor_is_plddt': True, 'sidecar_sha256': _digest(sidecar)}
    raise ValueError(f'{path.name}: local structure is not identified as AF2 with pLDDT in B-factors; '
                     'experimental B-factors are not pLDDT')


def _matching_chain(path, expected_sequence):
    try:
        from Bio.PDB import MMCIFParser, PDBParser
        from Bio.PDB.PDBExceptions import PDBConstructionException
    except ImportError as exc:
        raise RuntimeError('Biopython is required; install the fasta package extra') from exc
    import numpy as np
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in ('.cif', '.mmcif') else PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('input', str(path))
        models = list(structure.get_models())
        if len(models) != 1:
            raise ValueError('Exactly one structure model is required')
        matches = []
        observed_lengths = []
        for chain in models[0]:
            sequence, confidence = [], []
            for residue in chain:
                if residue.id[0] != ' ':
                    continue
                name = residue.get_resname().upper()
                if name not in _THREE_TO_ONE:
                    raise ValueError(f'Nonstandard protein residue {name} in chain {chain.id}')
                if residue.is_disordered():
                    raise ValueError('Disordered residue/alternate locations are unsupported')
                atoms = list(residue.get_atoms())
                if not atoms or any(atom.is_disordered() for atom in atoms):
                    raise ValueError('Missing or alternate-location atoms are unsupported')
                values = [float(atom.get_bfactor()) for atom in atoms]
                if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values):
                    raise ValueError('pLDDT values must be finite and in [0, 100]')
                sequence.append(_THREE_TO_ONE[name])
                confidence.append(float(np.mean(values)))
            observed = ''.join(sequence)
            if not observed:
                continue
            observed_lengths.append(len(observed))
            if observed == expected_sequence:
                chain_id = str(chain.id)
                if not re.fullmatch(r'[A-Za-z0-9]+', chain_id):
                    raise ValueError('Blank or special-character chain identifiers are unsupported')
                matches.append((chain_id, confidence))
        if len(matches) != 1:
            if matches:
                raise ValueError('Multiple chains exactly match FASTA; chain choice is ambiguous')
            raise ValueError(f'No complete structure chain exactly matches FASTA '
                             f'(FASTA length {len(expected_sequence)}, structure chain lengths {observed_lengths})')
        return matches[0]
    except (KeyError, TypeError, OSError, PDBConstructionException) as exc:
        raise ValueError(f'{path.name}: cannot parse a valid structure: {exc}') from exc


def _select_afdb_record(payload, protein, sequence):
    if not isinstance(payload, list) or not payload:
        raise ValueError(f'{protein}: no AlphaFold DB prediction records')
    matches = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        if (row.get('uniprotAccession') != protein or row.get('uniprotSequence') != sequence
                or row.get('sequence', sequence) != sequence):
            continue
        start = row.get('uniprotStart', row.get('sequenceStart'))
        end = row.get('uniprotEnd', row.get('sequenceEnd'))
        try:
            complete = int(start) == 1 and int(end) == len(sequence)
            version = int(row.get('latestVersion'))
        except (ValueError, TypeError):
            continue
        tool = row.get('toolUsed', '')
        entry = row.get('entryId', row.get('modelEntityId'))
        if (complete and version >= 1 and isinstance(tool, str) and _AF2.search(tool.upper())
                and entry == f'AF-{protein}-F1' and row.get('isComplex', False) is False):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f'{protein}: expected exactly one full-length AFDB record with the exact FASTA sequence; '
                         'fragments, isoform mismatches and ambiguous records are not silently selected')
    return matches[0]


def resolve_structures(records, workdir, structure_dir=None, source='alphafold', offline=False):
    """Resolve exact sequence-matched AF2 chains, returning ordered dictionaries.

    The caller's run directory may already exist; ``structures/`` must not.
    Local ID.pdb/ID.cif inputs are copied, never changed. With ``offline=True``
    or ``source='local'``, missing local files fail without making a request.
    """
    _validate_records(records)
    if source not in ('alphafold', 'local'):
        raise ValueError('Structure source must be alphafold or local')
    if structure_dir is not None:
        structure_dir = Path(structure_dir).expanduser().resolve(strict=True)
        if not structure_dir.is_dir():
            raise ValueError('structure-dir must be a directory')
    if (offline or source == 'local') and structure_dir is None:
        raise ValueError('Offline/local operation requires structure-dir')
    destination = _new_subdir(workdir, 'structures')
    resolved = []
    for protein, sequence in records:
        local = [] if structure_dir is None else [
            structure_dir / (protein + suffix) for suffix in ('.pdb', '.cif', '.mmcif')
            if (structure_dir / (protein + suffix)).is_file()]
        if len(local) > 1:
            raise ValueError(f'{protein}: multiple local structure files; provide only one')
        if local:
            identity = _af2_local_identity(local[0])
            path = destination / (protein + local[0].suffix.lower())
            with path.open('xb') as target, local[0].open('rb') as source_stream:
                shutil.copyfileobj(source_stream, target)
            chain, confidence = _matching_chain(path, sequence)
            provenance = {'source': 'local_alphafold', 'source_file': local[0].name, **identity}
        else:
            if offline or source == 'local':
                raise ValueError(f'{protein}: local AF2 structure is missing; offline mode forbids downloading')
            if not _UNIPROT.fullmatch(protein):
                raise ValueError(f'{protein}: AFDB retrieval requires a UniProt accession; '
                                 'provide a sequence-matched local AF2 structure for arbitrary identifiers')
            api_url = AFDB_API + quote(protein, safe='')
            raw = _fetch_bytes(api_url, api=True, max_bytes=20_000_000)
            try:
                row = _select_afdb_record(json.loads(raw), protein, sequence)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ValueError(f'{protein}: invalid AlphaFold DB metadata JSON') from exc
            with (destination / (protein + '.afdb.json')).open('xb') as target:
                target.write(raw)
            url = row.get('pdbUrl') or row.get('cifUrl')
            if not isinstance(url, str):
                raise ValueError(f'{protein}: AFDB did not provide a structure URL')
            _checked_url(url)
            suffix = Path(urlparse(url).path).suffix.lower()
            if suffix not in ('.pdb', '.cif', '.mmcif'):
                raise ValueError(f'{protein}: unsupported structure file format')
            entry_id = row.get('entryId', row.get('modelEntityId'))
            expected_name = f'{entry_id}-model_v{int(row["latestVersion"])}{suffix}'
            if Path(urlparse(url).path).name != expected_name:
                raise ValueError(f'{protein}: structure URL does not match AFDB entry/version metadata')
            content = _fetch_bytes(url)
            path = destination / (protein + suffix)
            with path.open('xb') as target:
                target.write(content)
            chain, confidence = _matching_chain(path, sequence)
            provenance = {'source': 'alphafold_database', 'accession': protein,
                          'entry_id': entry_id, 'version': int(row['latestVersion']),
                          'structure_method': row['toolUsed'],
                          'metadata_url': api_url, 'structure_url': url,
                          'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                          'bfactor_is_plddt': True}
        provenance.update(structure_sha256=_digest(path), sequence_sha256=_sequence_digest(sequence),
                          length=len(sequence), chain_id=chain, complete_sequence_match=True)
        resolved.append({'protein_id': protein, 'sequence': sequence, 'path': str(path),
                         'chain_id': chain, 'plddt': confidence, 'provenance': provenance})
    with (destination / 'provenance.json').open('x', encoding='utf-8') as target:
        json.dump([r['provenance'] for r in resolved], target, indent=2, allow_nan=False)
        target.write('\n')
    return resolved


def make_tokens(resolved, foldseek, workdir, plddt_threshold=70.0):
    """Return (AA+3Di records, provenance) without implicit trimming or skipping."""
    if not math.isfinite(plddt_threshold) or not 0 <= plddt_threshold <= 100:
        raise ValueError('pLDDT threshold must be finite and in [0, 100]')
    if not resolved:
        raise ValueError('No resolved structures were supplied')
    executable = shutil.which(str(foldseek)) or str(Path(foldseek).expanduser())
    binary = Path(executable).resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError('Foldseek must be an executable regular file')
    try:
        version = subprocess.run([str(binary), 'version'], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f'Cannot execute Foldseek on this platform: {exc}') from exc
    if version.returncode or not version.stdout.strip():
        raise ValueError('Foldseek version command failed; supply a compatible, versioned binary')
    destination = _new_subdir(workdir, 'tokens')
    records, rows = [], []
    for item in resolved:
        protein, sequence, chain = item['protein_id'], item['sequence'], item['chain_id']
        _validate_records([(protein, sequence)])
        confidence = item['plddt']
        if len(confidence) != len(sequence) or any(not math.isfinite(v) or not 0 <= v <= 100 for v in confidence):
            raise ValueError(f'{protein}: pLDDT coverage does not match the complete sequence')
        path = Path(item['path']).resolve(strict=True)
        if _digest(path) != item['provenance']['structure_sha256']:
            raise ValueError(f'{protein}: structure changed after sequence/confidence validation')
        with tempfile.TemporaryDirectory(prefix='foldseek-', dir=destination) as temporary:
            descriptor = Path(temporary) / 'descriptor.tsv'
            command = [str(binary), 'structureto3didescriptor', '-v', '0', '--threads', '1',
                       '--chain-name-mode', '1', str(path), str(descriptor)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ValueError(f'{protein}: Foldseek execution failed: {exc}') from exc
            if result.returncode or not descriptor.is_file():
                raise ValueError(f'{protein}: Foldseek descriptor generation failed '
                                 f'(exit {result.returncode}): {result.stderr[-1000:]}')
            matches = []
            for line in descriptor.read_text().splitlines():
                if not line.strip():
                    continue
                columns = line.split('\t')
                if len(columns) < 3:
                    raise ValueError(f'{protein}: malformed Foldseek descriptor row')
                name, amino_acids, structural = columns[:3]
                label = name.split()[0]
                if '_' not in label:
                    raise ValueError(f'{protein}: Foldseek output lacks explicit chain identifiers')
                prefix, result_chain = label.rsplit('_', 1)
                if result_chain == chain and Path(prefix).name in (path.name, path.stem):
                    matches.append((amino_acids, structural.lower()))
            if len(matches) != 1:
                raise ValueError(f'{protein}: Foldseek output did not identify exactly the selected structure chain')
            amino_acids, structural = matches[0]
            if amino_acids != sequence or len(structural) != len(sequence):
                raise ValueError(f'{protein}: Foldseek AA/3Di does not exactly cover the FASTA sequence')
            if set(structural) - STRUCTURE_ALPHABET:
                raise ValueError(f'{protein}: Foldseek returned unsupported 3Di symbols')
            masked = ''.join('#' if score < plddt_threshold else symbol
                             for symbol, score in zip(structural, confidence))
            combined = ''.join(aa + symbol for aa, symbol in zip(sequence, masked))
        records.append((protein, combined))
        rows.append({'protein_id': protein, 'length': len(sequence), 'chain_id': chain,
                     'structure_sha256': item['provenance']['structure_sha256'],
                     'token_sha256': hashlib.sha256(combined.encode('ascii')).hexdigest(),
                     'masked_residues': sum(score < plddt_threshold for score in confidence),
                     'plddt_threshold': plddt_threshold})
    provenance = {'foldseek_version': version.stdout.strip(), 'foldseek_sha256': _digest(binary),
                  'mode': 'structureto3didescriptor', 'chain_name_mode': 1, 'threads': 1,
                  'plddt_policy': 'mean per-residue atomic B-factor; below threshold masked as #',
                  'records': rows}
    with (destination / 'tokens.fasta').open('x', encoding='ascii') as target:
        for protein, combined in records:
            target.write(f'>{protein}\n{combined}\n')
    with (destination / 'provenance.json').open('x', encoding='utf-8') as target:
        json.dump(provenance, target, indent=2, allow_nan=False)
        target.write('\n')
    return records, provenance
