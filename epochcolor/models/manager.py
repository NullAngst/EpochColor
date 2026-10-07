"""The model manager: what can be downloaded, what is installed, and getting
weights from one to the other.

The catalog ships with the app and is refreshed from the GitHub repo, so a
new model shows up without an app update as long as EpochColor already has
an adapter for its architecture. Each installed model lives in its own
folder under the models directory, next to a manifest.json that says what
it is:

    models/<id>/manifest.json
    models/<id>/<weights file>

Downloads resume after an interruption and are checked against a SHA-256
before they are kept: the hash from the catalog when it has one, otherwise
the hash Hugging Face publishes for the file.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from .. import diskspace
from .weights import models_dir, sha256_file

CATALOG_URL = "https://raw.githubusercontent.com/NullAngst/EpochColor/HEAD/catalog.json"
ARCHITECTURES = ("eccv16", "siggraph17", "ddcolor")
MODES = ("automatic", "hint", "exemplar")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ModelError(RuntimeError):
    pass


# ------------------------------------------------------------------ catalog


def _bundled() -> list[dict]:
    return json.loads((Path(__file__).with_name("catalog.json")).read_text())["models"]


def _remote_cache() -> Path:
    return models_dir() / ".catalog-remote.json"


def validate_entry(e: dict) -> list[str]:
    errs = []
    for k in ("id", "name", "architecture", "mode", "license", "source"):
        if k not in e:
            errs.append(f"missing {k}")
    if "id" in e and not ID_RE.match(str(e["id"])):
        errs.append(f"bad id {e.get('id')!r}")
    if e.get("architecture") not in ARCHITECTURES:
        errs.append(f"no adapter for architecture {e.get('architecture')!r}")
    src = e.get("source", {})
    if src.get("type") not in ("url", "huggingface", "local"):
        errs.append("source type must be url, huggingface or local")
    return errs


def catalog() -> list[dict]:
    """The bundled catalog, with the last fetched remote one layered on top.
    Entries for architectures this build can't run are left out."""
    by_id = {e["id"]: e for e in _bundled()}
    try:
        remote = json.loads(_remote_cache().read_text()).get("models", [])
    except (OSError, ValueError):
        remote = []
    for e in remote:
        if isinstance(e, dict) and not validate_entry(e):
            by_id[e["id"]] = e
    return list(by_id.values())


def refresh_catalog(timeout: float = 20) -> tuple[int, int]:
    """Fetch the catalog from the repo. Returns (usable entries, skipped)."""
    req = urllib.request.Request(CATALOG_URL, headers={"User-Agent": "EpochColor"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    models = data.get("models", []) if isinstance(data, dict) else []
    good = [e for e in models if isinstance(e, dict) and not validate_entry(e)]
    _remote_cache().parent.mkdir(parents=True, exist_ok=True)
    _remote_cache().write_text(json.dumps({"fetched": time.time(), "models": good}))
    return len(good), len(models) - len(good)


def entry(model_id: str) -> dict | None:
    return next((e for e in catalog() if e["id"] == model_id), None)


# ---------------------------------------------------------------- installed


def installed() -> dict[str, dict]:
    """id -> manifest, for every model with its weights present."""
    out: dict[str, dict] = {}
    root = models_dir()
    if root.is_dir():
        for m in root.glob("*/manifest.json"):
            try:
                man = json.loads(m.read_text())
            except (OSError, ValueError):
                continue
            w = m.parent / man.get("weights", "")
            if man.get("id") and w.is_file():
                man["_path"] = str(w)
                out[man["id"]] = man
    # weights downloaded by earlier versions sit loose in the folder
    for e in _bundled():
        src = e["source"]
        if e["id"] not in out and src.get("filename") and (root / src["filename"]).is_file():
            out[e["id"]] = {**e, "weights": src["filename"], "_path": str(root / src["filename"])}
    return out


def resolve(model_id: str) -> tuple[dict, Path]:
    man = installed().get(model_id)
    if man is None:
        if entry(model_id):
            raise ModelError(f"{model_id} is not downloaded. Download it in Colour > Model manager, "
                             f"or run: epochcolor fetch {model_id}")
        raise ModelError(f"unknown model {model_id!r}")
    return man, Path(man["_path"])


def remove(model_id: str) -> bool:
    man = installed().get(model_id)
    if man is None:
        return False
    p = Path(man["_path"])
    if p.parent != models_dir():
        shutil.rmtree(p.parent)
    else:
        p.unlink()  # a loose legacy file
    return True


# ----------------------------------------------------------------- download


def _get(url: str, timeout: float = 60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "EpochColor"}),
                                  timeout=timeout)


def _hf_file(repo: str) -> tuple[str, str, str | None, int | None]:
    """(url, filename, sha256, size) of the weights in a Hugging Face repo."""
    with _get(f"https://huggingface.co/api/models/{repo}/tree/main") as r:
        files = json.loads(r.read().decode())
    names = {f["path"]: f for f in files if f.get("type") == "file"}
    prefer = ["model.safetensors", "pytorch_model.safetensors", "pytorch_model.bin", "pytorch_model.pt"]
    pick = next((n for n in prefer if n in names), None)
    if pick is None:
        cands = [n for n in names if n.endswith((".safetensors", ".pth", ".pt", ".bin"))]
        if not cands:
            raise ModelError(f"no weights file found in {repo}")
        pick = max(cands, key=lambda n: names[n].get("size", 0))
    f = names[pick]
    sha = (f.get("lfs") or {}).get("oid")
    return f"https://huggingface.co/{repo}/resolve/main/{pick}", pick, sha, f.get("size")


def download(model_id: str, progress=None, accept_license: bool = False) -> Path:
    """Download a catalog model into its own folder. progress(stage, done, total)."""
    e = entry(model_id)
    if e is None:
        raise ModelError(f"{model_id} is not in the catalog")
    if not e.get("license_ok") and not accept_license:
        raise ModelError(f"{model_id}'s license needs accepting first: {e['license']} "
                         f"({e.get('license_url', '')}). Pass --accept-license once you've read it.")
    src = e["source"]
    sha_full, sha_prefix = src.get("sha256"), src.get("sha256_prefix")
    if src["type"] == "url":
        url, fname = src["url"], src["filename"]
    elif src["type"] == "huggingface":
        url, fname, hf_sha, _ = _hf_file(src["repo"])
        sha_full = sha_full or hf_sha
    else:
        raise ModelError(f"{model_id} is a local model and can't be downloaded")
    if not sha_full and not sha_prefix:
        raise ModelError(f"no hash to check {model_id} against, refusing to keep it")

    folder = models_dir() / model_id
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / fname
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "EpochColor"})
    if src.get("size_mb") or e.get("size_mb"):
        need = int((src.get("size_mb") or e["size_mb"]) * 1.05e6) - have
        diskspace.need(folder, need, f"{model_id}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:
            have = 0  # the server ignored the range; start over
        size = resp.headers.get("Content-Length")
        total = int(size) + have if size else None
        if total:
            diskspace.need(folder, total - have, f"{model_id}")
        try:
            with open(part, "ab" if have else "wb") as f:
                done = have
                last = 0.0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress and time.time() - last > 0.3:
                        progress("download", done, total or done)
                        last = time.time()
        except OSError as ex:
            if diskspace.is_full(ex):
                raise ModelError(diskspace.explain(ex, part) +
                                 " The partial download is kept and resumes next time.") from None
            raise
    digest = sha256_file(part, progress)
    ok = digest == sha_full if sha_full else digest.startswith(sha_prefix)
    if not ok:
        part.unlink()
        raise ModelError(f"{fname} failed its SHA-256 check, download deleted")
    part.rename(dest)
    man = {k: v for k, v in e.items() if not k.startswith("_")}
    man.update({"weights": fname, "sha256": digest, "installed": time.time(),
                "license_accepted": bool(accept_license and not e.get("license_ok"))})
    (folder / "manifest.json").write_text(json.dumps(man, indent=1))
    if progress:
        progress("download", 1, 1)
    return dest


# ----------------------------------------------------------------- add file


MANIFEST_HELP = """\
A manifest is a small JSON file next to the weights:
  {
    "id": "my-model",                 lowercase, digits, dot, dash, underscore
    "name": "My fine-tuned DDColor",
    "architecture": "ddcolor",        one of: eccv16, siggraph17, ddcolor
    "mode": "automatic",              automatic or hint
    "license": "CC-BY-NC-4.0",
    "params": {"model_size": "large", "input_size": 512},
    "color_space": "lab"              EpochColor adapters all take L and give a/b
  }"""


def add_from_file(weights: str | Path, manifest: dict | str | Path) -> str:
    """Install a weights file the user has, described by a manifest. The
    weights are copied in; the original stays where it is."""
    weights = Path(weights)
    if not weights.is_file():
        raise ModelError(f"no such file: {weights}")
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    man = dict(manifest)
    man.setdefault("mode", "automatic")
    man.setdefault("params", {})
    man.setdefault("license", "unknown")
    man["source"] = {"type": "local", "original": str(weights)}
    errs = validate_entry(man)
    if man.get("color_space", "lab") != "lab":
        errs.append("only lab colour space models have adapters")
    if errs:
        raise ModelError("manifest problems: " + "; ".join(errs) + "\n\n" + MANIFEST_HELP)
    if man["id"] in installed():
        raise ModelError(f"a model called {man['id']} is installed already; remove it first")
    folder = models_dir() / man["id"]
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, folder / weights.name)
    man.update({"weights": weights.name, "sha256": sha256_file(folder / weights.name),
                "installed": time.time(), "license_ok": True})
    (folder / "manifest.json").write_text(json.dumps(man, indent=1))
    return man["id"]


def check_loads(model_id: str, device: str = "cpu") -> None:
    """Build the adapter and load the weights, so a bad file fails here
    instead of halfway through a colorize."""
    from . import create

    m = create(model_id)
    m.load(device)


def move_storage(new_dir: str | Path | None) -> Path:
    """Point the models directory somewhere else (None: back to the default),
    moving what's installed along."""
    from .weights import default_models_dir, write_setting

    old = models_dir()
    new = Path(new_dir).expanduser() if new_dir else default_models_dir()
    if new.resolve() == old.resolve():
        return new
    new.mkdir(parents=True, exist_ok=True)
    if old.is_dir():
        for item in old.iterdir():
            target = new / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
    write_setting("models_dir", str(new) if new_dir else None)
    return new


def human_size(mb) -> str:
    if not mb:
        return "?"
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{int(mb)} MB"


if __name__ == "__main__":  # quick listing for debugging
    for e in catalog():
        print(e["id"], "installed" if e["id"] in installed() else "", file=sys.stderr)
