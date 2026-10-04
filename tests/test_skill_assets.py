import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import pytest
from PhysicalRSI_baselines.robodojo.pack_skill_assets import pack

path = Path(__file__).parents[1] / 'PhysicalRSI_baselines/robodojo/releases/robodojo-v1/download_assets.py'
spec = importlib.util.spec_from_file_location('skill_download', path)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_asset_parts_roundtrip_and_checksum(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'weights').write_bytes(bytes(range(256)) * 50)
    cache = tmp_path / 'release'
    asset = pack(source, cache, 'test', cache.as_uri(), 'checkpoints/test', part_bytes=1024)
    assert len(asset['parts']) > 1
    target = download.install(asset, tmp_path / 'installed')
    assert (target / 'weights').read_bytes() == (source / 'weights').read_bytes()
    with pytest.raises(FileExistsError):
        download.install(asset, tmp_path / 'installed')
    part = cache / asset['parts'][0]['name']
    part.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        download.install(asset, tmp_path / 'other')


def test_asset_archive_cannot_escape_destination(tmp_path):
    archive = tmp_path / 'bad.tar'
    with tarfile.open(archive, 'w') as stream:
        member = tarfile.TarInfo('../outside')
        member.size = 1
        stream.addfile(member, io.BytesIO(b'x'))
    asset = dict(format='tar-parts', destination='safe', parts=[dict(name='bad.tar', url=archive.as_uri(),
                 size=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())])
    with pytest.raises(ValueError):
        download.install(asset, tmp_path / 'install')
    assert not (tmp_path / 'install/outside').exists()


def test_deduplicated_archive_restores_every_path_once(tmp_path):
    from PhysicalRSI_baselines.robodojo.pack_skill_assets import pack_deduplicated
    source = tmp_path / 'source'
    (source / 'first').mkdir(parents=True)
    (source / 'second').mkdir()
    (source / 'empty-required-directory').mkdir()
    data = bytes(range(256)) * 10
    (source / 'first/weights').write_bytes(data)
    (source / 'second/weights').write_bytes(data)
    cache = tmp_path / 'release'
    asset = pack_deduplicated(source, cache, 'shared', cache.as_uri(), 'skills/code')
    assert asset['files'] == 2
    assert asset['unique_blobs'] == 1
    target = download.install(asset, tmp_path / 'installed')
    assert (target / 'empty-required-directory').is_dir()
    assert (target / 'first/weights').read_bytes() == data
    assert (target / 'second/weights').read_bytes() == data
    assert (target / 'first/weights').stat().st_ino == (target / 'second/weights').stat().st_ino
