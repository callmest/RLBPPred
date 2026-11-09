"""Explicit, hash-pinned downloads for the frozen inference release.

Downloads never happen during prediction without an explicit setup request.
Only authenticated files are published; corrupt caches and symlinks are refused
rather than overwritten. Large files are streamed, not loaded into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform as host_platform
import shutil
import stat
import sys
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .encoder import CHECKPOINT_BYTES, CHECKPOINT_REVISION, CHECKPOINT_SHA256


@dataclass(frozen=True)
class Asset:
    filename: str
    url: str
    size: int
    sha256: str


_RELEASE = "https://github.com/callmest/RLBPPred-inference/releases/download/v0.2.0"
ASSETS = {
    "saprot": Asset(
        "SaProt_650M_AF2.pt",
        "https://huggingface.co/westlake-repl/SaProt_650M_AF2/resolve/"
        f"{CHECKPOINT_REVISION}/SaProt_650M_AF2.pt",
        CHECKPOINT_BYTES, CHECKPOINT_SHA256,
    ),
    "mlp-weights": Asset(
        "model.safetensors", f"{_RELEASE}/model.safetensors", 949_122_160,
        "e9f2db20805d6212606851a29e69b8d286ba443b8ae0e2f22872060e963ac3b5",
    ),
    "mlp-manifest": Asset(
        "model.json", f"{_RELEASE}/model.json", 1012,
        "82458d98d41b694861c24911a8554564e59a49adf471eb034a44d8da94d9c247",
    ),
    "foldseek-darwin": Asset(
        "foldseek-osx-universal-10-941cd33.tar.gz",
        "https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-osx-universal.tar.gz",
        17_021_264, "6d8b07e188d443044f0c98db30aa3d4cca6750e97b50679372b54b45a664cae5",
    ),
    "foldseek-linux-avx2": Asset(
        "foldseek-linux-avx2-10-941cd33.tar.gz",
        "https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-avx2.tar.gz",
        11_941_946, "e6a3e3860e76b0b7b858736f627fe680d4d7f0fc7298301d8a92b9f711d74ac7",
    ),
}
_FOLDSEEK_BINARIES = {
    "foldseek-darwin": (41_955_896, "1b446bbece6b01cf0a50b28419f353522204f44a10a5872462f8ad400d9aadab"),
    "foldseek-linux-avx2": (29_402_960, "4db94a7825d409e416643772860ddf7cfe328d582679b47e6dffda7df4d41ea0"),
}


def _https(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Model assets may only be downloaded over HTTPS without URL credentials")


class _HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        _https(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def _directory(path):
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError(f"Cache directory must not be a symbolic link: {requested}")
    requested.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not requested.is_dir():
        raise ValueError(f"Expected a directory: {requested}")
    return requested.resolve()


def _validate_file(path, asset):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Cached {asset.filename} is not a regular, non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if before.st_size != asset.size:
            raise ValueError(f"Cached {asset.filename} has the wrong size; move it aside before retrying")
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"Cached {asset.filename} changed during verification")
    if digest.hexdigest() != asset.sha256:
        raise ValueError(f"Cached {asset.filename} has the wrong SHA-256; move it aside before retrying")
    return path


def _publish_file(temporary, destination):
    # Hard-link publication is atomic and refuses even a concurrently created
    # destination. Both files live in the same directory/filesystem.
    os.link(temporary, destination)


def download_asset(name, cache_dir, *, timeout=60, retries=2, progress=True):
    """Explicitly download one named pinned asset, or verify and reuse its cache.

    Recognized names are the keys of ``ASSETS``. Unknown names, partial or corrupt
    existing files, and symlinks fail; this function never silently repairs or
    overwrites an existing cache entry. ``retries`` counts additional attempts
    after transient HTTPS transport/server failures, not integrity failures.
    """
    if name not in ASSETS:
        raise ValueError(f"Unknown asset {name!r}; choose from {', '.join(sorted(ASSETS))}")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
        raise ValueError("timeout must be greater than zero and at most 600 seconds")
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 5:
        raise ValueError("retries must be an integer between zero and five")
    asset = ASSETS[name]
    _https(asset.url)
    directory = _directory(cache_dir)
    destination = directory / asset.filename
    if destination.exists() or destination.is_symlink():
        return _validate_file(destination, asset)
    opener = build_opener(_HTTPSRedirect())
    for attempt in range(retries + 1):
        temporary = None
        try:
            request = Request(asset.url, headers={"User-Agent": "RLBPPred/0.2.0", "Accept-Encoding": "identity"})
            with opener.open(request, timeout=timeout) as response:
                _https(response.geturl())
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != asset.size:
                    raise ValueError(f"{name}: server Content-Length does not match the pinned size")
                digest, total, last_report = hashlib.sha256(), 0, 0
                with tempfile.NamedTemporaryFile(mode="w+b", prefix=".asset-", dir=directory,
                                                  delete=False) as handle:
                    temporary = Path(handle.name)
                    if progress:
                        print(f"Downloading {name} ({asset.size:,} bytes)", file=sys.stderr)
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        total += len(block)
                        if total > asset.size:
                            raise ValueError(f"{name}: download exceeded the pinned byte size")
                        handle.write(block)
                        digest.update(block)
                        if progress and total - last_report >= 128 * 1024 * 1024:
                            print(f"{name}: {total:,}/{asset.size:,} bytes", file=sys.stderr)
                            last_report = total
                    if total != asset.size or digest.hexdigest() != asset.sha256:
                        raise ValueError(f"{name}: downloaded size or SHA-256 mismatch; no asset was published")
                    handle.flush()
                    os.fsync(handle.fileno())
            _publish_file(temporary, destination)
            if progress:
                print(f"Verified {name}", file=sys.stderr)
            return destination
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            # Do not retry exclusive-publication conflicts or permanent HTTP
            # errors (in particular an unreleased GitHub asset's 404).
            retryable = not isinstance(exc, (FileExistsError, PermissionError))
            if isinstance(exc, HTTPError):
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt == retries:
                raise RuntimeError(f"Cannot download {name}: {exc}") from exc
            if progress:
                print(f"Retrying {name} after a transient download failure", file=sys.stderr)
            time.sleep(min(attempt + 1, 2))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Cannot download {name}")  # Defensive; loop always returns or raises.


def download_saprot(cache_dir, **kwargs):
    """Return the verified fixed-revision official SaProt-AF2 checkpoint."""
    return download_asset("saprot", cache_dir, **kwargs)


def _publish_directory(source, destination):
    """Atomically rename a directory without replacing even an empty target."""
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes, destination_bytes = os.fsencode(source), os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source_bytes, destination_bytes, 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100, source_bytes, -100, destination_bytes, 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif os.name == "nt":
        # Windows os.rename already refuses an existing destination.
        os.rename(source, destination)
        return
    else:
        raise RuntimeError("This platform does not support atomic non-replacing model-directory publication")
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(error, os.strerror(error), str(destination))
        raise OSError(error, os.strerror(error), str(destination))


def _copy_verified(source, destination, asset):
    digest, size = hashlib.sha256(), 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            size += len(block)
            if size > asset.size:
                raise ValueError(f"{asset.filename} cache changed during copying")
            writer.write(block)
            digest.update(block)
        writer.flush()
        os.fsync(writer.fileno())
    if size != asset.size or digest.hexdigest() != asset.sha256:
        raise ValueError(f"{asset.filename} cache changed during copying")


def download_model(model_dir, *, cache_dir=None, **kwargs):
    """Install only the two verified strict-MLP release files into a new directory.

    An already complete, exact two-file directory is verified and reused. Other
    existing targets are refused. New bundles are assembled separately and then
    published atomically, without ever replacing an existing directory.
    """
    requested = Path(model_dir).expanduser()
    if requested.is_symlink():
        raise ValueError("Model directory must not be a symbolic link")
    expected = [ASSETS["mlp-weights"], ASSETS["mlp-manifest"]]
    if requested.exists():
        if not requested.is_dir() or {path.name for path in requested.iterdir()} != {asset.filename for asset in expected}:
            raise FileExistsError("Model target exists and is not an exact two-file inference bundle")
        for asset in expected:
            _validate_file(requested / asset.filename, asset)
        return requested.resolve()
    cache = cache_dir if cache_dir is not None else Path.home() / ".cache" / "rlbppred" / "assets"
    sources = [download_asset(name, cache, **kwargs) for name in ("mlp-weights", "mlp-manifest")]
    requested.parent.mkdir(parents=True, exist_ok=True)
    destination = requested.resolve()
    staging = Path(tempfile.mkdtemp(prefix=".rlbppred-model-", dir=destination.parent))
    try:
        for source, asset in zip(sources, expected):
            _copy_verified(source, staging / asset.filename, asset)
        _publish_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def _foldseek_platform(platform, machine):
    system = host_platform.system().lower() if platform is None else str(platform).lower()
    architecture = host_platform.machine().lower() if machine is None else str(machine).lower()
    if system == "darwin" and architecture in {"arm64", "x86_64"}:
        return "foldseek-darwin"
    if system == "linux" and architecture == "x86_64":
        return "foldseek-linux-avx2"
    raise ValueError("Pinned Foldseek binaries support Darwin arm64/x86_64 and Linux x86_64 (AVX2) only")


def _extract_foldseek(archive_path, output, binary_asset):
    """Manually copy one regular member; never use tar extract/extractall."""
    with tarfile.open(archive_path, "r:gz") as archive:
        matches = []
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("Foldseek archive contains an unsafe member path")
            if not (member.isdir() or member.isfile()):
                raise ValueError("Foldseek archive links and special-file members are not accepted")
            if str(path) == "foldseek/bin/foldseek":
                if not member.isfile():
                    raise ValueError("Foldseek binary archive member must be a regular file")
                matches.append(member)
        if len(matches) != 1 or matches[0].size != binary_asset.size:
            raise ValueError("Foldseek archive must contain one correctly sized foldseek/bin/foldseek member")
        reader = archive.extractfile(matches[0])
        if reader is None:
            raise ValueError("Foldseek binary cannot be read from its authenticated archive")
        digest, size = hashlib.sha256(), 0
        output.parent.mkdir(parents=True)
        with reader, output.open("xb") as writer:
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                size += len(block)
                if size > binary_asset.size:
                    raise ValueError("Foldseek archive binary exceeded its pinned size")
                writer.write(block)
                digest.update(block)
            writer.flush()
            os.fsync(writer.fileno())
        if size != binary_asset.size or digest.hexdigest() != binary_asset.sha256:
            raise ValueError("Foldseek archive binary SHA-256 or size mismatch")
        output.chmod(0o755)


def download_foldseek(cache_dir, *, platform=None, machine=None, **kwargs):
    """Install a pinned Foldseek binary, returning cache_dir/foldseek/bin/foldseek.

    macOS arm64/x86_64 uses the universal release; Linux x86_64 uses AVX2.
    Unsupported platforms fail explicitly. Only the authenticated regular tar
    member ``foldseek/bin/foldseek`` is copied, with mode 755. Provenance records
    both the pinned archive and extracted-binary digests. No links, arbitrary
    paths, archive scripts, or install hooks are extracted or executed.
    """
    name = _foldseek_platform(platform, machine)
    archive_asset = ASSETS[name]
    binary_size, binary_hash = _FOLDSEEK_BINARIES[name]
    binary_asset = Asset("foldseek", archive_asset.url, binary_size, binary_hash)
    root = _directory(cache_dir)
    destination = root / "foldseek"
    binary = destination / "bin" / "foldseek"
    metadata = {
        "schema_version": 1,
        "tool": "foldseek",
        "version": "10-941cd33",
        "platform_asset": name,
        "archive_url": archive_asset.url,
        "archive_bytes": archive_asset.size,
        "archive_sha256": archive_asset.sha256,
        "archive_member": "foldseek/bin/foldseek",
        "binary_bytes": binary_size,
        "binary_sha256": binary_hash,
    }
    if destination.is_symlink():
        raise ValueError("Foldseek cache directory must not be a symbolic link")
    if destination.exists():
        if not destination.is_dir() or (destination / "bin").is_symlink():
            raise ValueError("Foldseek cache is not a regular installation directory")
        _validate_file(binary, binary_asset)
        receipt = destination / "asset.json"
        if receipt.is_symlink() or not receipt.is_file():
            raise ValueError("Foldseek cache is missing its regular provenance receipt; move it aside before retrying")
        if receipt.stat().st_size > 16_384 or json.loads(receipt.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Foldseek cache provenance does not match the pinned platform release")
        if stat.S_IMODE(binary.stat().st_mode) != 0o755:
            raise ValueError("Cached Foldseek binary must retain its executable mode 755")
        return binary
    archive_path = download_asset(name, root / "archives", **kwargs)
    staging = Path(tempfile.mkdtemp(prefix=".foldseek-", dir=root))
    try:
        _extract_foldseek(archive_path, staging / "bin" / "foldseek", binary_asset)
        with (staging / "asset.json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return binary
