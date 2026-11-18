"""Make the small Chinese fonts the game ships with.

  pip install fonttools
  python3 web/tools/make_font.py /path/to/NotoSerifSC[wght].ttf

Takes the Noto Serif SC variable font (SIL Open Font License, from
github.com/notofonts/noto-cjk), pins it to weight 400 and 700, and keeps only
the characters used in web/godot/scripts/*.gd plus ASCII. Run again after
adding new Chinese text.
"""
import sys
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

GODOT = Path(__file__).resolve().parents[1] / "godot"
OUT = GODOT / "assets" / "fonts"


def chars() -> str:
    text = "".join(p.read_text(encoding="utf-8") for p in (GODOT / "scripts").glob("*.gd"))
    keep = {c for c in text if ord(c) > 127}
    keep |= {chr(i) for i in range(32, 127)}
    keep |= set("—·　「」（）…，。：！？％✓×◉♪")
    return "".join(sorted(keep))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    text = chars()
    for weight, name in [(400, "serif.ttf"), (700, "serif-bold.ttf")]:
        font = instancer.instantiateVariableFont(TTFont(src), {"wght": weight})
        opts = subset.Options()
        opts.layout_features = ["*"]
        opts.name_IDs = ["*"]
        sub = subset.Subsetter(opts)
        sub.populate(text=text)
        sub.subset(font)
        cmap = font.getBestCmap()
        missing = [c for c in text if ord(c) not in cmap and c not in " \t"]
        font.save(OUT / name)
        print(f"{name}: {len(cmap)} glyphs, {(OUT / name).stat().st_size // 1024} KB"
              + (f", missing {''.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
