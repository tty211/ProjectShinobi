"""Build the web game into web/dist/, ready for Cloudflare Pages.

  python3 web/tools/build.py [--godot /path/to/Godot] [--skip-godot]

Steps:
  1. Godot exports web/godot as the "Web" preset into dist/ (index.html,
     index.js, index.wasm, index.pck, audio worklets).
  2. Files over Cloudflare's 25 MiB limit are gzipped (index.wasm and the
     ONNX Runtime wasm). The page shell unpacks them in the browser.
  3. Camera code, ONNX Runtime, models, songs and voice lines are copied in.
Only the Python standard library is used.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
DIST = WEB / "dist"
LIMIT = 25 * 1024 * 1024

# Song files get plain ASCII names on the site.
SONGS = {
    "128bpm踊.mp3": "odoru.mp3",
    "135bpm紅蓮華.mp3": "gurenge.mp3",
    "136bpmil vento d'oro.mp3": "vento.mp3",
}

HEADERS = """/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Resource-Policy: same-origin
  Permissions-Policy: camera=(self), microphone=(), geolocation=()
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer
  X-Frame-Options: DENY

/vision/models/*
  Cache-Control: public, max-age=604800

/vision/ort/*
  Cache-Control: public, max-age=604800

/music/*
  Cache-Control: public, max-age=604800
"""


def find_godot(arg):
    for cand in [arg, os.environ.get("GODOT"),
                 str(ROOT.parent.parent / ".tools/godot/Godot.app/Contents/MacOS/Godot"),
                 "/Applications/Godot.app/Contents/MacOS/Godot", shutil.which("godot")]:
        if cand and Path(cand).exists():
            return cand
    sys.exit("Godot 4.7 not found: pass --godot /path/to/Godot or set GODOT")


def gz(src: Path, dst: Path):
    with open(src, "rb") as f, open(dst, "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as g:
        shutil.copyfileobj(f, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--godot")
    ap.add_argument("--skip-godot", action="store_true")
    args = ap.parse_args()

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()

    if not args.skip_godot:
        godot = find_godot(args.godot)
        subprocess.run([godot, "--headless", "--path", str(WEB / "godot"), "--import"], check=True)
        subprocess.run([godot, "--headless", "--path", str(WEB / "godot"),
                        "--export-release", "Web", str(DIST / "index.html")], check=True)
        wasm = DIST / "index.wasm"
        if wasm.exists() and wasm.stat().st_size > LIMIT:
            gz(wasm, DIST / "index.wasm.gz")
            wasm.unlink()
        for f in DIST.iterdir():
            if f.stat().st_size > LIMIT:
                sys.exit(f"{f.name} is over 25 MiB; Cloudflare Pages will refuse it")

    # Camera + recognition.
    vision = DIST / "vision"
    (vision / "ort").mkdir(parents=True)
    (vision / "models").mkdir()
    shutil.copy(WEB / "vision" / "vision.js", vision)
    shutil.copy(WEB / "vision" / "bench.html", vision)
    ort_src = WEB / "vision" / "ort"
    for f in ort_src.iterdir():
        if f.suffix == ".wasm":
            gz(f, vision / "ort" / (f.name + ".gz"))
        else:
            shutil.copy(f, vision / "ort")
    models = sorted((WEB / "vision" / "models").iterdir())
    for f in models:
        shutil.copy(f, vision / "models")
    manifest = {
        "ortWasm": "ort/ort-wasm-simd-threaded.asyncify.wasm.gz",
        "hand": [f.name for f in models if f.name.startswith("hand.onnx")],
        "gesture": [f.name for f in models if f.name.startswith("gesture.onnx")],
    }
    manifest["totalBytes"] = sum(
        (vision / p).stat().st_size
        for p in [manifest["ortWasm"]] + ["models/" + n for n in manifest["hand"] + manifest["gesture"]])
    (vision / "models" / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # Sound.
    (DIST / "music").mkdir()
    for src, dst in SONGS.items():
        shutil.copy(ROOT / "music" / src, DIST / "music" / dst)
    shutil.copytree(ROOT / "audio", DIST / "audio")

    (DIST / "_headers").write_text(HEADERS)

    total = 0
    for f in sorted(DIST.rglob("*")):
        if f.is_file():
            total += f.stat().st_size
            if f.stat().st_size > LIMIT:
                sys.exit(f"{f} is over 25 MiB")
    print(f"dist ready: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
