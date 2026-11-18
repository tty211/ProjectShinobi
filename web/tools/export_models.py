"""Turn best.pt (hand finder) and best_gesture.pt (seal reader) into the
browser models in web/vision/models/.

  python web/tools/export_models.py

Needs: pip install ultralytics onnx onnxslim

Each model is exported to ONNX with dynamic input size, then its weights are
stored as float16 with a Cast back to float32 in front of every use. The
graph still computes in float32 (ONNX Runtime folds the Casts when it loads),
so answers match the .pt files while the download is half the size.
Cloudflare Pages refuses files over 25 MiB, so the seal model is cut into
parts that vision.js joins again.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "vision" / "models"
PART_BYTES = 20 * 1024 * 1024


def export(pt: Path, work: Path) -> Path:
    local = work / pt.name
    shutil.copy(pt, local)
    return Path(YOLO(str(local)).export(format="onnx", dynamic=True, simplify=True, opset=17, imgsz=640))


def halve(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    inits, casts = [], []
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= 256:
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "__f16")
            inits.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT))
        else:
            inits.append(init)
    del graph.initializer[:]
    graph.initializer.extend(inits)
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model


def write_parts(data: bytes, stem: str) -> None:
    for old in OUT.glob(stem + ".onnx*"):
        old.unlink()
    if len(data) <= PART_BYTES:
        (OUT / f"{stem}.onnx").write_bytes(data)
        return
    for i in range(0, len(data), PART_BYTES):
        (OUT / f"{stem}.onnx.part{i // PART_BYTES}").write_bytes(data[i:i + PART_BYTES])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for pt, stem in ((ROOT / "best.pt", "hand"), (ROOT / "best_gesture.pt", "gesture")):
            model = halve(onnx.load(str(export(pt, Path(tmp)))))
            write_parts(model.SerializeToString(), stem)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:24} {f.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
