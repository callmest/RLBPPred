"""Tiny mocked HTTPS responses exercise downloads without network or models."""

import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from rlbppred import assets


@pytest.mark.parametrize("name,filename", [("mlp-weights", "model.safetensors"),
                                           ("mlp-manifest", "model.json")])
def test_frozen_model_urls_use_main_software_repository(name, filename):
    assert assets.ASSETS[name].url == (
        "https://github.com/callmest/RLBPPred/releases/download/v0.2.0/" + filename
    )


class Response(io.BytesIO):
    def __init__(self, data, url="https://example.org/file", content_length=None):
        super().__init__(data)
        self.url = url
        self.headers = {"Content-Length": str(len(data) if content_length is None else content_length)}

    def geturl(self):
        return self.url


@pytest.fixture
def tiny_assets(monkeypatch):
    data = {"saprot": b"tiny-saprot", "mlp-weights": b"tiny-weights", "mlp-manifest": b'{"tiny": true}\n'}
    filenames = {"saprot": "SaProt_650M_AF2.pt", "mlp-weights": "model.safetensors", "mlp-manifest": "model.json"}
    descriptions = {
        name: assets.Asset(filenames[name], f"https://example.org/{name}", len(value), hashlib.sha256(value).hexdigest())
        for name, value in data.items()
    }
    monkeypatch.setattr(assets, "ASSETS", descriptions)
    calls = []

    def open_response(request, timeout):
        calls.append((request, timeout))
        name = request.full_url.rsplit("/", 1)[1]
        return Response(data[name], url=request.full_url)

    monkeypatch.setattr(assets, "build_opener", lambda handler: SimpleNamespace(open=open_response))
    monkeypatch.setattr(assets.time, "sleep", lambda delay: None)
    return data, descriptions, calls


def test_download_then_verified_cache_reuse(tmp_path, tiny_assets):
    data, descriptions, calls = tiny_assets
    path = assets.download_saprot(tmp_path, progress=False)
    assert path == tmp_path / descriptions["saprot"].filename
    assert path.read_bytes() == data["saprot"]
    assert len(calls) == 1
    assert assets.download_saprot(tmp_path, progress=False) == path
    assert len(calls) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["SaProt_650M_AF2.pt"]
    assert calls[0][0].get_header("Accept-encoding") == "identity"


@pytest.mark.parametrize("damage", [b"short", b"bad-bad-bad"])
def test_corrupt_existing_cache_is_not_repaired(tmp_path, tiny_assets, damage):
    _, descriptions, calls = tiny_assets
    path = tmp_path / descriptions["saprot"].filename
    path.write_bytes(damage)
    with pytest.raises(ValueError, match="wrong size|wrong SHA-256"):
        assets.download_saprot(tmp_path, progress=False)
    assert path.read_bytes() == damage and calls == []


def test_symlink_cache_and_symlink_directory_refused(tmp_path, tiny_assets):
    _, descriptions, calls = tiny_assets
    symlink = tmp_path / descriptions["saprot"].filename
    symlink.symlink_to(tmp_path / "nonexistent")
    with pytest.raises(ValueError, match="not a regular"):
        assets.download_saprot(tmp_path, progress=False)
    linked_dir = tmp_path / "linked-dir"
    linked_dir.symlink_to(tmp_path / "new-cache")
    with pytest.raises(ValueError, match="symbolic link"):
        assets.download_saprot(linked_dir, progress=False)
    assert symlink.is_symlink() and calls == []


@pytest.mark.parametrize("kind", ["hash", "length", "oversized", "truncated", "http"])
def test_integrity_failure_does_not_publish(tmp_path, tiny_assets, monkeypatch, kind):
    data, descriptions, _ = tiny_assets
    expected = data["saprot"]
    response = {
        "hash": Response(b"x" * len(expected)),
        "length": Response(expected, content_length=3),
        "oversized": Response(expected + b"x", content_length=len(expected)),
        "truncated": Response(expected[:-1], content_length=len(expected)),
        "http": Response(expected, url="http://example.org/file"),
    }[kind]
    monkeypatch.setattr(assets, "build_opener", lambda handler: SimpleNamespace(open=lambda *a, **k: response))
    with pytest.raises(ValueError, match="SHA-256|Content-Length|byte size|HTTPS"):
        assets.download_saprot(tmp_path, progress=False)
    assert not list(tmp_path.iterdir())


def test_retry_transient_error_only(tmp_path, tiny_assets, monkeypatch):
    data, _, _ = tiny_assets
    calls = []

    def open_response(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise URLError("temporary network failure")
        return Response(data["saprot"])
    monkeypatch.setattr(assets, "build_opener", lambda handler: SimpleNamespace(open=open_response))
    assert assets.download_saprot(tmp_path, progress=False).read_bytes() == data["saprot"]
    assert len(calls) == 3


def test_404_is_not_retried(tmp_path, tiny_assets, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.org/file", 404, "not found", {}, None)
    monkeypatch.setattr(assets, "build_opener", lambda handler: SimpleNamespace(open=fail))
    with pytest.raises(RuntimeError, match="Cannot download"):
        assets.download_saprot(tmp_path, progress=False)
    assert len(calls) == 1 and not list(tmp_path.iterdir())


def test_rejects_https_to_http_redirect_before_following():
    handler = assets._HTTPSRedirect()
    request = Request("https://example.org/start")
    with pytest.raises(ValueError, match="HTTPS"):
        handler.redirect_request(request, None, 302, "redirect", {}, "http://example.org/end")


@pytest.mark.parametrize("kwargs", [{"retries": -1}, {"retries": True}, {"retries": 6},
                                    {"timeout": 0}, {"timeout": True}, {"timeout": 601}])
def test_invalid_download_options(tmp_path, tiny_assets, kwargs):
    with pytest.raises(ValueError):
        assets.download_saprot(tmp_path, progress=False, **kwargs)


def test_unknown_asset(tmp_path):
    with pytest.raises(ValueError, match="Unknown asset"):
        assets.download_asset("other-model", tmp_path)


def test_model_installs_exact_two_files_and_reuses_verified(tmp_path, tiny_assets):
    data, _, calls = tiny_assets
    output, cache = tmp_path / "model", tmp_path / "cache"
    assert assets.download_model(output, cache_dir=cache, progress=False) == output
    assert sorted(p.name for p in output.iterdir()) == ["model.json", "model.safetensors"]
    assert (output / "model.json").read_bytes() == data["mlp-manifest"]
    assert (output / "model.safetensors").read_bytes() == data["mlp-weights"]
    assert len(calls) == 2
    assert assets.download_model(output, cache_dir=cache, progress=False) == output
    assert len(calls) == 2


def test_model_target_existing_partial_or_extra_files_refused(tmp_path, tiny_assets):
    output = tmp_path / "model"
    output.mkdir()
    (output / "user-file").write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="exact two-file"):
        assets.download_model(output, cache_dir=tmp_path / "cache", progress=False)
    assert (output / "user-file").read_bytes() == b"preserve"


def test_model_publication_failure_preserves_target_and_cleans_stage(tmp_path, tiny_assets, monkeypatch):
    output, cache = tmp_path / "model", tmp_path / "cache"

    def concurrent_directory(source, destination):
        output.mkdir()
        (output / "user").write_bytes(b"preserve")
        raise FileExistsError("concurrent target")
    monkeypatch.setattr(assets, "_publish_directory", concurrent_directory)
    with pytest.raises(FileExistsError):
        assets.download_model(output, cache_dir=cache, progress=False)
    assert (output / "user").read_bytes() == b"preserve"
    assert not list(tmp_path.glob(".rlbppred-model-*"))


def test_atomic_directory_never_replaces_existing_empty_directory(tmp_path):
    source, destination = tmp_path / "staging", tmp_path / "existing"
    source.mkdir()
    destination.mkdir()
    before = destination.stat().st_ino
    with pytest.raises(FileExistsError):
        assets._publish_directory(source, destination)
    assert source.exists() and destination.stat().st_ino == before


def test_concurrent_asset_is_not_overwritten(tmp_path, tiny_assets, monkeypatch):
    output = tmp_path / "SaProt_650M_AF2.pt"

    def publish(temporary, destination):
        output.write_bytes(b"user-file")
        raise FileExistsError("concurrent file")
    monkeypatch.setattr(assets, "_publish_file", publish)
    with pytest.raises(RuntimeError, match="concurrent file"):
        assets.download_saprot(tmp_path, progress=False)
    assert output.read_bytes() == b"user-file"
    assert sorted(p.name for p in tmp_path.iterdir()) == [output.name]


def _tar_bytes(binary, extra=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("foldseek/bin/foldseek")
        member.size = len(binary)
        archive.addfile(member, io.BytesIO(binary))
        if extra is not None:
            member = tarfile.TarInfo(extra[0])
            member.type = extra[1]
            if member.issym() or member.islnk():
                member.linkname = "../../outside"
            archive.addfile(member)
    return buffer.getvalue()


@pytest.fixture
def tiny_foldseek(tmp_path, monkeypatch):
    binary = b"tiny-pinned-foldseek-binary"
    archive = _tar_bytes(binary)
    names = ["foldseek-darwin", "foldseek-linux-avx2"]
    descriptions = {
        name: assets.Asset(name + ".tar.gz", "https://example.org/" + name,
                           len(archive), hashlib.sha256(archive).hexdigest())
        for name in names
    }
    monkeypatch.setattr(assets, "ASSETS", descriptions)
    monkeypatch.setattr(assets, "_FOLDSEEK_BINARIES", {
        name: (len(binary), hashlib.sha256(binary).hexdigest()) for name in names
    })
    calls = []

    def open_response(request, timeout):
        calls.append(request)
        return Response(archive, url=request.full_url)
    monkeypatch.setattr(assets, "build_opener", lambda handler: SimpleNamespace(open=open_response))
    return binary, archive, descriptions, calls


@pytest.mark.parametrize("platform,machine", [("Darwin", "arm64"), ("darwin", "x86_64"),
                                              ("Linux", "x86_64")])
def test_foldseek_download_extract_receipt_and_reuse(tmp_path, tiny_foldseek, platform, machine):
    binary, _, _, calls = tiny_foldseek
    path = assets.download_foldseek(tmp_path, platform=platform, machine=machine, progress=False)
    assert path == tmp_path / "foldseek" / "bin" / "foldseek"
    assert path.read_bytes() == binary
    assert stat.S_IMODE(path.stat().st_mode) == 0o755
    receipt = json.loads((path.parent.parent / "asset.json").read_text())
    assert receipt["archive_member"] == "foldseek/bin/foldseek"
    assert receipt["binary_sha256"] == hashlib.sha256(binary).hexdigest()
    assert assets.download_foldseek(tmp_path, platform=platform, machine=machine, progress=False) == path
    assert len(calls) == 1


@pytest.mark.parametrize("platform,machine", [("windows", "AMD64"), ("linux", "aarch64"),
                                              ("darwin", "mips"), ("other", "x86_64")])
def test_foldseek_unsupported_platform_no_network(tmp_path, tiny_foldseek, platform, machine):
    with pytest.raises(ValueError, match="support Darwin"):
        assets.download_foldseek(tmp_path, platform=platform, machine=machine, progress=False)
    assert not tiny_foldseek[-1] and not list(tmp_path.iterdir())


@pytest.mark.parametrize("name,member_type", [("../outside", tarfile.REGTYPE),
                                              ("/absolute", tarfile.REGTYPE),
                                              ("foldseek/../../outside", tarfile.REGTYPE),
                                              ("foldseek\\outside", tarfile.REGTYPE),
                                              ("foldseek/link", tarfile.SYMTYPE),
                                              ("foldseek/link", tarfile.LNKTYPE),
                                              ("foldseek/pipe", tarfile.FIFOTYPE),
                                              ("foldseek/bin/foldseek", tarfile.REGTYPE)])
def test_foldseek_tar_rejects_unsafe_or_duplicate_members(tmp_path, name, member_type):
    binary = b"test-binary"
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(_tar_bytes(binary, extra=(name, member_type)))
    description = assets.Asset("foldseek", "https://example.org/foldseek", len(binary),
                               hashlib.sha256(binary).hexdigest())
    with pytest.raises(ValueError, match="unsafe|links|one correctly"):
        assets._extract_foldseek(archive, tmp_path / "safe" / "bin" / "foldseek", description)
    assert not (tmp_path / "safe").exists()


def test_foldseek_wrong_binary_hash_never_publishes(tmp_path, tiny_foldseek, monkeypatch):
    binary, _, _, _ = tiny_foldseek
    monkeypatch.setattr(assets, "_FOLDSEEK_BINARIES", {"foldseek-darwin": (len(binary), "0" * 64)})
    with pytest.raises(ValueError, match="binary SHA-256"):
        assets.download_foldseek(tmp_path, platform="darwin", machine="arm64", progress=False)
    assert not (tmp_path / "foldseek").exists()
    assert not list(tmp_path.glob(".foldseek-*"))


def test_foldseek_existing_corruption_not_overwritten(tmp_path, tiny_foldseek):
    path = assets.download_foldseek(tmp_path, platform="darwin", machine="arm64", progress=False)
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="wrong size"):
        assets.download_foldseek(tmp_path, platform="darwin", machine="arm64", progress=False)
    assert path.read_bytes() == b"corrupted"


def test_foldseek_symlink_or_provenance_change_refused(tmp_path, tiny_foldseek):
    path = assets.download_foldseek(tmp_path, platform="darwin", machine="arm64", progress=False)
    receipt = path.parent.parent / "asset.json"
    metadata = json.loads(receipt.read_text())
    metadata["version"] = "different"
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance does not match"):
        assets.download_foldseek(tmp_path, platform="darwin", machine="arm64", progress=False)
    receipt.unlink()
    receipt.symlink_to(tmp_path / "other")
    with pytest.raises(ValueError, match="regular provenance receipt"):
        assets.download_foldseek(tmp_path, platform="darwin", machine="arm64", progress=False)
