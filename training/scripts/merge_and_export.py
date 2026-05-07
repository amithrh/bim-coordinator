"""Merge LoRA adapter into base model and export for deployment.

Outputs three deployment-ready artifacts:
  1. fused/         - MLX-format merged model (fast local inference on Mac)
  2. hf/            - HuggingFace safetensors format (cross-platform)
  3. gguf/          - Quantized GGUF for Ollama / llama.cpp / Cloud Run

The HF and GGUF stages are optional (--skip-hf, --skip-gguf) since GGUF
conversion requires llama.cpp which is a separate install.

Usage:
  python merge_and_export.py --adapter <adapter_path>
  python merge_and_export.py --adapter <adapter_path> --quantize Q4_K_M
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _find_llamacpp() -> Path | None:
    """Look in common spots for llama.cpp's convert script.

    Returns a path that contains both convert_hf_to_gguf.py and llama-quantize.
    Supports: source checkouts, /opt/llama.cpp, brew install (Cellar).
    """
    candidates = [
        Path.home() / "src" / "llama.cpp",
        Path.home() / "code" / "llama.cpp",
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
    ]
    # Brew install: /opt/homebrew/Cellar/llama.cpp/<version>/bin
    cellar = Path("/opt/homebrew/Cellar/llama.cpp")
    if cellar.exists():
        for ver in sorted(cellar.iterdir(), reverse=True):
            if (ver / "bin" / "convert_hf_to_gguf.py").exists():
                candidates.append(ver / "bin")  # both tools live in /bin
    # /opt/homebrew/bin symlinks
    if Path("/opt/homebrew/bin/convert_hf_to_gguf.py").exists():
        candidates.append(Path("/opt/homebrew/bin"))

    for c in candidates:
        if (c / "convert_hf_to_gguf.py").exists():
            return c
    return None


def _find_quantize_bin(llamacpp_dir: Path) -> Path | None:
    """Locate the llama-quantize binary given a llama.cpp dir."""
    for path in [
        llamacpp_dir / "build" / "bin" / "llama-quantize",
        llamacpp_dir / "llama-quantize",
        llamacpp_dir / "build_x86" / "bin" / "llama-quantize",
        llamacpp_dir / "bin" / "llama-quantize",
        Path("/opt/homebrew/bin/llama-quantize"),
    ]:
        if path.exists():
            return path
    return None


def _read_manifest(adapter_path: Path) -> dict:
    mf = adapter_path / "manifest.json"
    if not mf.exists():
        raise SystemExit(f"No manifest.json in {adapter_path} - was training completed?")
    return json.loads(mf.read_text())


def fuse_adapter(adapter_path: Path, base_model: str, output_dir: Path) -> None:
    """Use mlx_lm fuse to merge LoRA into base, saving in MLX format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", base_model,
        "--adapter-path", str(adapter_path),
        "--save-path", str(output_dir),
        "--de-quantize",   # produce bf16 weights for downstream conversion
    ]
    print(f"[fuse] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def export_hf(fused_dir: Path, hf_dir: Path) -> None:
    """Re-save the fused model in HuggingFace format using transformers.

    For Gemma 4, the fused MLX dir is already mostly HF-compatible (config.json,
    tokenizer files, model-00001-of-N.safetensors). We just normalise it.
    """
    hf_dir.mkdir(parents=True, exist_ok=True)
    # Copy everything; mlx fuse already writes safetensors + tokenizer + config
    for item in fused_dir.iterdir():
        target = hf_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    print(f"[export_hf] HF-format model at {hf_dir}")


def convert_to_gguf(
    hf_dir: Path,
    gguf_path: Path,
    llamacpp_dir: Path,
    quant_type: str = "Q4_K_M",
) -> None:
    """Use llama.cpp to convert HF safetensors -> GGUF, then quantize."""
    gguf_path.parent.mkdir(parents=True, exist_ok=True)

    convert_script = llamacpp_dir / "convert_hf_to_gguf.py"
    intermediate_gguf = gguf_path.with_suffix(".f16.gguf")

    # Step 1: HF -> GGUF (fp16)
    cmd = [
        sys.executable, str(convert_script),
        str(hf_dir),
        "--outfile", str(intermediate_gguf),
        "--outtype", "f16",
    ]
    print(f"[convert_to_gguf] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Step 2: Quantize
    quantize_bin = _find_quantize_bin(llamacpp_dir)
    if not quantize_bin:
        raise SystemExit(
            f"llama-quantize not found. Install with: brew install llama.cpp"
        )

    cmd = [str(quantize_bin), str(intermediate_gguf), str(gguf_path), quant_type]
    print(f"[quantize] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Cleanup intermediate
    intermediate_gguf.unlink()
    size_mb = gguf_path.stat().st_size / 1024 / 1024
    print(f"[done] {gguf_path} ({size_mb:.1f} MB, {quant_type})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True,
                        help="Path to a trained LoRA adapter directory")
    parser.add_argument("--output-base", default="training/exports",
                        help="Where to put fused/, hf/, gguf/ subdirs")
    parser.add_argument("--quantize", default="Q4_K_M",
                        help="GGUF quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--llamacpp", help="Path to llama.cpp checkout")
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    adapter_path = Path(args.adapter).resolve()
    if not adapter_path.exists():
        print(f"Adapter path does not exist: {adapter_path}", file=sys.stderr)
        return 2

    manifest = _read_manifest(adapter_path)
    base_model = manifest["base_model"]
    run_name = manifest["run_name"]

    out_base = (repo_root / args.output_base / run_name).resolve()
    fused_dir = out_base / "fused"
    hf_dir = out_base / "hf"
    gguf_dir = out_base / "gguf"
    gguf_path = gguf_dir / f"{run_name}.{args.quantize}.gguf"

    print(f"\n=== Merge + Export ===")
    print(f"  adapter:    {adapter_path}")
    print(f"  base model: {base_model}")
    print(f"  output:     {out_base}\n")

    fuse_adapter(adapter_path, base_model, fused_dir)

    if args.skip_hf:
        print("[skip] HF export")
    else:
        export_hf(fused_dir, hf_dir)

    if args.skip_gguf:
        print("[skip] GGUF export")
    else:
        llamacpp = Path(args.llamacpp) if args.llamacpp else _find_llamacpp()
        if not llamacpp:
            print(
                "[warn] llama.cpp not found. Skipping GGUF.\n"
                "       Install with: git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp\n"
                "                     && cd ~/llama.cpp && cmake -B build && cmake --build build -j",
                file=sys.stderr,
            )
        else:
            convert_to_gguf(hf_dir, gguf_path, llamacpp, args.quantize)

    # Update manifest with export paths
    manifest["exports"] = {
        "fused_mlx": str(fused_dir),
        "hf": str(hf_dir) if not args.skip_hf else None,
        "gguf": str(gguf_path) if not args.skip_gguf else None,
        "quantization": args.quantize,
    }
    (adapter_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nUpdated manifest: {adapter_path / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
