"""
BANKAI CLI - Command Line Interface
====================================

All CLI functionality: banners, argument parsing, command dispatch.
Invoked via:
    $ bankai-run [command] [options]
    $ python -m bankai [command] [options]
"""

import argparse
import os
import random
import sys
from typing import Optional

from bankai import GPU_AVAILABLE, GPU_MEMORY, GPU_NAME, __version__, get_gpu_info

# ===============================
# Banner System
# ===============================


def _should_show_banner() -> bool:
    """バナーを表示すべきか判定"""
    if os.environ.get("BANKAI_NO_BANNER"):
        return False
    if not sys.stdout.isatty():
        return False
    if "--no-banner" in sys.argv:
        return False
    return True


def _gpu_status_line() -> str:
    """GPU状態の1行サマリ"""
    if GPU_AVAILABLE:
        return f"GPU: {GPU_NAME} ({GPU_MEMORY:.1f} GB)"
    return "CPU Mode (install CuPy for GPU acceleration)"


def print_banner(style: Optional[str] = None):
    """バナー表示（スタイル指定可）"""
    if not _should_show_banner():
        return

    style = style or os.environ.get("BANKAI_BANNER_STYLE", "random").lower()

    banners = {
        "simple": _banner_simple,
        "ascii": _banner_ascii,
        "matrix": _banner_matrix,
        "tamaki": _banner_tamaki,
    }

    if style == "random":
        random.choice(list(banners.values()))()
    elif style in banners:
        banners[style]()
    else:
        _banner_simple()


def _banner_simple():
    """シンプルバナー"""
    print()
    print("=" * 60)
    print(f"  BANKAI v{__version__}")
    print("  Bond-vector ANalysis of Kinetic Amino acid Initiator")
    print("=" * 60)
    print(f"  {_gpu_status_line()}")
    print("=" * 60)
    print()


def _banner_ascii():
    """ASCIIアートバナー"""
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██████   █████  ███    ██ ██   ██  █████  ██           ║
║   ██   ██ ██   ██ ████   ██ ██  ██  ██   ██ ██           ║
║   ██████  ███████ ██ ██  ██ █████   ███████ ██           ║
║   ██   ██ ██   ██ ██  ██ ██ ██  ██  ██   ██ ██           ║
║   ██████  ██   ██ ██   ████ ██   ██ ██   ██ ██           ║
║                                                           ║
║   Bond-vector ANalysis of Kinetic Amino acid Initiator    ║
║   v{__version__:<53s}║
║                                                           ║
╠═══════════════════════════════════════════════════════════╣
║   {_gpu_status_line():<55s}║
╚═══════════════════════════════════════════════════════════╝
""")


def _banner_matrix():
    """マトリックス風バナー"""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  01000010 01000001 01001110 01001011 01000001 01001001   ║
║           B  A  N  K  A  I  //  SYSTEM ONLINE           ║
║  Causal cascade detection engine v{__version__:<22s}║
╚══════════════════════════════════════════════════════════╝
  [{_gpu_status_line()}]
""")


def _banner_tamaki():
    """環ちゃんバナー"""
    faces = ["(◕‿◕)", "(｡♥‿♥｡)", "(✧ω✧)", "(*´▽`*)"]
    messages = [
        "BANKAI起動したよ〜！",
        "今日も解析頑張ろ〜！",
        "ご主人さま、準備OK！",
        "カスケード見つけちゃうぞ〜！",
    ]
    face = random.choice(faces)
    msg = random.choice(messages)

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   BANKAI v{__version__:<48s}║
║   Bond-vector ANalysis of Kinetic Amino acid Initiator    ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║   {face} < {msg:<43s}║
║                                                           ║
║   {_gpu_status_line():<55s}║
║   Powered by Miosync, Inc.                                ║
╚═══════════════════════════════════════════════════════════╝
""")


# ===============================
# CLI Commands
# ===============================


def cmd_analyze(args):
    """解析実行コマンド"""
    print("\n🚀 Starting BANKAI analysis...")
    print(f"   Trajectory: {args.trajectory}")
    print(f"   Metadata:   {args.metadata}")
    print(f"   Output:     {args.output}")
    if GPU_AVAILABLE:
        print(f"   GPU Device: {args.device}")
    else:
        print("   Mode:       CPU")
    print()

    from bankai.analysis.run_full_analysis import main as run_analysis

    # run_full_analysisに引数を渡す
    sys.argv = [
        "bankai-run",
        args.trajectory,
        args.metadata,
        "--output",
        args.output,
    ]
    if args.device is not None:
        sys.argv.extend(["--device", str(args.device)])
    run_analysis()


def cmd_benchmark(args):
    """ベンチマーク実行"""
    print("\n⚡ Running BANKAI GPU benchmark...\n")

    if not GPU_AVAILABLE:
        print("❌ No GPU detected. Benchmark requires GPU.")
        sys.exit(1)

    import time as _time

    import cupy as cp

    size = 10000
    a = cp.random.rand(size, size, dtype=cp.float32)
    b = cp.random.rand(size, size, dtype=cp.float32)

    # Warm up
    cp.dot(a, b)
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    start = _time.time()
    n_iter = 10
    for _ in range(n_iter):
        cp.dot(a, b)
    cp.cuda.Stream.null.synchronize()
    elapsed = _time.time() - start

    gflops = (n_iter * 2 * size**3) / (elapsed * 1e9)

    print(f"   GPU:        {GPU_NAME}")
    print(f"   Memory:     {GPU_MEMORY:.1f} GB")
    print(f"   Matrix:     {size} x {size} (float32)")
    print(f"   Iterations: {n_iter}")
    print(f"   Time:       {elapsed:.3f} s")
    print(f"   Throughput: {gflops:.1f} GFLOPS")
    print("\n✨ Benchmark complete!")


def cmd_info(args):
    """システム情報表示"""
    info = get_gpu_info()

    print("\n📊 BANKAI System Information")
    print(f"   Version:     {__version__}")
    print(f"   GPU:         {'Available' if info['available'] else 'Not available'}")
    if info["available"]:
        print(f"   Device:      {info['name']}")
        print(f"   Memory:      {info['memory_gb']:.1f} GB")
        print(f"   CUDA:        {info['cuda_version']}")
        print(f"   Compute:     {info['compute_capability']}")
    print(f"   CuPy:        {'Installed' if info['has_cupy'] else 'Not installed'}")
    print(f"   Python:      {sys.version.split()[0]}")
    print()


def cmd_check_gpu(args):
    """GPU動作確認"""
    if not GPU_AVAILABLE:
        print("❌ GPU not available")
        print("   Install CuPy: pip install bankai[cuda12]")
        sys.exit(1)

    print(f"✅ GPU OK: {GPU_NAME} ({GPU_MEMORY:.1f} GB)")

    # 簡易テスト
    try:
        import cupy as cp

        x = cp.ones(1000)
        assert float(cp.sum(x)) == 1000.0
        print("✅ CuPy computation OK")
    except Exception as e:
        print(f"❌ CuPy test failed: {e}")
        sys.exit(1)


# ===============================
# Argument Parser
# ===============================


def build_parser() -> argparse.ArgumentParser:
    """CLIパーサーを構築"""
    parser = argparse.ArgumentParser(
        prog="bankai",
        description=(
            "BANKAI: Bond-vector ANalysis of Kinetic Amino acid Initiator\n"
            "GPU-accelerated sub-picosecond causal cascade detection "
            "in GROMACS trajectories."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bankai analyze trajectory.npy metadata.json --output ./results
  bankai benchmark
  bankai info
  bankai check-gpu

GitHub: https://github.com/miosync-masa/bankai
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"BANKAI v{__version__}",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress startup banner",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Run full BANKAI analysis")
    p_analyze.add_argument("trajectory", help="Trajectory file (.npy)")
    p_analyze.add_argument("metadata", help="Metadata file (.json)")
    p_analyze.add_argument(
        "--output",
        "-o",
        default="./results",
        help="Output directory (default: ./results)",
    )
    p_analyze.add_argument(
        "--device", type=int, default=0, help="GPU device ID (default: 0)"
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Run GPU benchmark")
    p_bench.set_defaults(func=cmd_benchmark)

    # --- info ---
    p_info = sub.add_parser("info", help="Show system information")
    p_info.set_defaults(func=cmd_info)

    # --- check-gpu ---
    p_gpu = sub.add_parser("check-gpu", help="Verify GPU availability")
    p_gpu.set_defaults(func=cmd_check_gpu)

    return parser


# ===============================
# Main Entry Point
# ===============================


def main():
    """CLI メインエントリーポイント"""
    print_banner()

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n💡 Try: bankai analyze --help")
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
