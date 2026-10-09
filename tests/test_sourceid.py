"""Caches follow a source's content, not its path or modification time."""

import os

from epochcolor.sourceid import CHUNK, adopt, source_id


def test_same_content_same_id_anywhere(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(os.urandom(3 * CHUNK))
    (tmp_path / "Disk1").mkdir()
    b = tmp_path / "Disk1" / "renamed.bin"
    b.write_bytes(a.read_bytes())
    assert source_id(a) == source_id(b)
    os.utime(a, (1_000_000_000, 1_000_000_000))
    assert source_id(a) == source_id(b), "a new modification time isn't a new picture"
    b.write_bytes(a.read_bytes()[:-1] + b"x")
    assert source_id(a) != source_id(b)


def test_big_file_reads_samples(tmp_path):
    big = tmp_path / "big.bin"
    data = bytearray(os.urandom(20 * CHUNK))
    big.write_bytes(data)
    one = source_id(big)
    data[3 * CHUNK] ^= 1  # between the samples
    big.write_bytes(data)
    os.utime(big, (2_000_000_000, 2_000_000_000))
    assert source_id(big) == one, "only the samples count, so a 200 GB file is quick"
    data[10 * CHUNK] ^= 1  # the middle sample
    big.write_bytes(data)
    os.utime(big, (2_000_000_001, 2_000_000_001))
    assert source_id(big) != one


def test_adopt(tmp_path):
    old, new = tmp_path / "old", tmp_path / "sub" / "new"
    old.mkdir()
    (old / "f").write_text("x")
    assert adopt(old, new) and (new / "f").read_text() == "x" and not old.exists()
    assert not adopt(old, new), "nothing left to move"


def test_photo_result_adopts_old_name(tmp_path, monkeypatch):
    import hashlib
    import json

    from epochcolor.worker import photo_result_path

    monkeypatch.setenv("EPOCHCOLOR_CACHE", str(tmp_path / "cache"))
    src = tmp_path / "ph.png"
    src.write_bytes(os.urandom(1000))
    st = src.stat()
    settings = {"working_size": 512, "grain": 100.0, "denoise": None}
    legacy = {"p": str(src.resolve()), "s": st.st_size, "m": st.st_mtime_ns,
              "set": settings, "model": "eccv16", "hints": None}
    old = tmp_path / "cache" / "photos" / (hashlib.sha256(json.dumps(legacy, sort_keys=True).encode())
                                           .hexdigest()[:20] + ".png")
    old.parent.mkdir(parents=True)
    old.write_bytes(b"png")
    got = photo_result_path(str(src), settings, "eccv16")
    assert got.exists() and got.read_bytes() == b"png" and not old.exists()
