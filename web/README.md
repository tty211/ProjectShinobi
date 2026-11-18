# Project Shinobi: web version

A Godot 4.7 version of `ninjutsu-v10.html` that anyone can play in a browser.
The hand-seal models run in the player's browser (ONNX Runtime Web, WebGPU with
a WASM fallback), so no Python server or NVIDIA GPU is needed. The camera
video never leaves the browser.

Live: https://jutsu-prototype.pages.dev (Cloudflare Pages project `jutsu-prototype`).

## Folders

| Path | What it is |
|---|---|
| `godot/` | The Godot project: `scripts/main.gd` (the game), `scripts/data.gd` (seals, jutsus, songs, Chinese + English text), `scripts/vision.gd` (talks to the camera code), `shell.html` (web page around the game) |
| `vision/vision.js` | Camera + recognition: finds the hand with `best.pt`, crops it, reads the seal with `best_gesture.pt` (same steps and thresholds as `server.py`) |
| `vision/models/` | The two models as ONNX (fp16 weights). `gesture.onnx` is split into 20 MiB parts because Cloudflare Pages takes files up to 25 MiB |
| `vision/ort/` | ONNX Runtime Web 1.31.1 (MIT licence, github.com/microsoft/onnxruntime) |
| `vision/bench.html` | Speed check: `bench.html?img=a.jpg&runs=5` |
| `tools/build.py` | Builds everything into `dist/` |
| `tools/export_models.py` | Remakes `vision/models/` from `best.pt` and `best_gesture.pt` |
| `tools/make_font.py` | Remakes the small Noto Serif SC fonts after adding Chinese text |
| `tools/serve.py` | Local test server with the headers the site needs |

## Build and test

Needs Godot 4.7.2 with web export templates, and Python 3.

```sh
python3 web/tools/build.py --godot /path/to/Godot   # → web/dist/
python3 web/tools/serve.py 8060                      # http://127.0.0.1:8060/
```

Add `?debug` to the address to play without a camera: keys 1–9, 0, -, = are
the 12 seals (子 丑 寅 卯 辰 巳 午 未 申 酉 戌 亥) and Space is always the right one.

## Deploy

```sh
npx wrangler pages deploy web/dist --project-name jutsu-prototype --branch main
```

`dist/_headers` turns the camera on (`Permissions-Policy: camera=(self)`) and
sets the cross-origin isolation headers ONNX Runtime needs for threads.

## Game rules (same as ninjutsu-v10.html)

- A seal counts when the model is at least 40 % sure for 2 checks in a row.
- Practice: no timer. Challenge: 10 s per seal, or the jutsu starts over.
- Music: one seal per bar (4 beats at the song's BPM), 5 jutsus a game; a
  jutsu passes with at most one missed seal.

## Remaking the models

```sh
pip install ultralytics onnx onnxslim
python3 web/tools/export_models.py
```

## Fonts

`godot/assets/fonts/` is a subset of Noto Serif SC (SIL Open Font License 1.1).
After adding new Chinese text, get `NotoSerifSC[wght].ttf` from
github.com/notofonts/noto-cjk and run `python3 web/tools/make_font.py <that file>`.
