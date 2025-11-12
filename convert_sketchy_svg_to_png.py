# convert_sketchy_svg_to_png.py
import argparse, os, re, sys, multiprocessing as mp
from pathlib import Path

# pip install cairosvg==2.7.0
import cairosvg

SVG_RE = re.compile(rb"<svg[\s\S]*</svg>", re.IGNORECASE)

def to_png(in_path: Path, out_path: Path, size: int = 256):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():  # resume-friendly
        return True, "skip"

    data = in_path.read_bytes()
    # If it's an HTML wrapper, extract the <svg>…</svg>
    if in_path.suffix.lower() == ".html":
        m = SVG_RE.search(data)
        if not m:
            return False, "no_svg_tag"
        data = m.group(0)

    try:
        cairosvg.svg2png(bytestring=data,
                         write_to=str(out_path),
                         output_width=size,
                         output_height=size,
                         background_color="white")
        return True, "ok"
    except Exception as e:
        return False, f"render_error: {e}"

def worker(args):
    in_file, in_root, out_root, size = args
    rel = in_file.relative_to(in_root)
    # change extension to .png
    out_file = (out_root / rel).with_suffix(".png")
    ok, msg = to_png(in_file, out_file, size=size)
    return ok, msg, str(in_file), str(out_file)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="sketchy/sketches",
                    help="Root dir containing class folders with .html/.svg files")
    ap.add_argument("--out_root", default="sketchy_png256",
                    help="Output root for rendered PNGs (mirrors folder structure)")
    ap.add_argument("--size", type=int, default=256, help="PNG size (square)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()-1))
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not in_root.exists():
        print(f"✗ Input root not found: {in_root}", file=sys.stderr)
        sys.exit(1)

    candidates = list(in_root.rglob("*.html")) + list(in_root.rglob("*.svg"))
    if not candidates:
        print(f"✗ No .html/.svg files under {in_root}", file=sys.stderr)
        sys.exit(1)

    print(f"• Found {len(candidates)} sketches")
    jobs = [(p, in_root, out_root, args.size) for p in candidates]

    ok_count = 0
    skip_count = 0
    fail = []

    with mp.Pool(args.workers) as pool:
        for ok, msg, src, dst in pool.imap_unordered(worker, jobs, chunksize=32):
            if ok and msg == "ok":
                ok_count += 1
            elif ok and msg == "skip":
                skip_count += 1
            else:
                fail.append((src, msg))

    print(f"✓ Rendered: {ok_count}")
    print(f"• Skipped (already exists): {skip_count}")
    if fail:
        print(f"✗ Failed: {len(fail)}")
        for src, why in fail[:20]:
            print("  -", src, "->", why)
        if len(fail) > 20:
            print(f"  … and {len(fail)-20} more")

if __name__ == "__main__":
    main()

'''
python convert_sketchy_svg_to_png.py --in_root sketchy/sketches --out_root sketchy_png256  --size 256  --workers 8
'''