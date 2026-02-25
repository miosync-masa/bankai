"""
bankai- Setup Script
High-performance GPU implementation of Lambda³ framework
Updated for CUDA 12.x compatibility
"""

from setuptools import setup, find_packages
import os
import sys
import subprocess

# バージョン情報
VERSION = '1.1.0'  # バージョンアップ！

# READMEを読み込む
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# CUDAバージョンの自動検出（改良版）
def get_cuda_version():
    """インストールされているCUDAバージョンを検出"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            # バージョン番号を抽出
            import re
            match = re.search(r'release (\d+)\.(\d+)', output)
            if match:
                major, minor = match.groups()
                return f"{major}.{minor}", int(major), int(minor)
    except:
        pass
    return None, None, None

# CUDA環境に応じた依存関係を設定
cuda_version_str, cuda_major, cuda_minor = get_cuda_version()

# 基本の依存関係
install_requires = [
    'numpy>=1.20.0,<2.0.0',
    # CuPyはここで動的に設定
    'scipy>=1.7.0,<2.0.0',
    'scikit-learn>=1.0.0,<2.0.0',
    'pandas>=1.3.0,<2.0.0',
    'matplotlib>=3.4.0,<4.0.0',
    'seaborn>=0.11.0,<1.0.0',
    'plotly>=5.0.0,<6.0.0',
    'psutil>=5.8.0,<6.0.0',
    'GPUtil>=1.4.0,<2.0.0',
    'networkx>=2.6.0,<4.0.0',
    'joblib>=1.0.0,<2.0.0',
    # 新しい依存関係
    'xarray>=2023.0.0',
    'h5py>=3.0.0',
    'netCDF4>=1.6.0',
    'zarr>=2.13.0',
]

# CUDAバージョンに応じてパッケージを選択
if cuda_version_str:
    print(f"🔍 Detected CUDA version: {cuda_version_str}")
    
    if cuda_major == 11:
        # CUDA 11.x
        install_requires.extend([
            'cupy-cuda11x>=10.0.0,<14.0.0',
            'numba>=0.56.0,<0.60.0',
        ])
    elif cuda_major == 12:
        if cuda_minor >= 5:
            # CUDA 12.5 - 特別な対応が必要
            print("⚠️  CUDA 12.5 detected - using compatibility mode")
            install_requires.extend([
                'cupy-cuda12x==13.2.0',  # 12.4対応版を固定
                'numba==0.59.1',         # 互換性のあるバージョン
            ])
        else:
            # CUDA 12.0-12.4
            install_requires.extend([
                'cupy-cuda12x>=13.0.0,<14.0.0',
                'numba>=0.59.0,<0.60.0',
            ])
    else:
        print(f"⚠️  CUDA {cuda_version_str} is not officially supported")
        install_requires.extend([
            'cupy>=10.0.0',  # 汎用版
            'numba>=0.56.0',
        ])
else:
    print("⚠️  CUDA not detected. GPU features may not work.")
    install_requires.extend([
        'cupy>=10.0.0',  # 汎用版（ユーザーが適切なものを選ぶ）
        'numba>=0.56.0',
    ])

# 開発用依存関係
extras_require = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
        'black>=21.0',
        'flake8>=3.9.0',
        'mypy>=0.910',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'ipdb>=0.13.0',
    ],
    'viz': [
        'jupyterlab>=3.0.0',
        'ipywidgets>=7.6.0',
        'notebook>=6.4.0',
    ],
    'cuda12': [
        # CUDA 12.x用の追加パッケージ
        'nvidia-cuda-nvjitlink-cu12',
        'cuda-python>=12.0.0',
    ]
}

# プラットフォーム別の設定
if sys.platform == 'win32':
    # Windows固有の設定
    pass
elif sys.platform == 'darwin':
    # macOS - GPUサポートなし
    print("⚠️  macOS detected - GPU acceleration not available")

# setup関数
setup(
    name='bankai',
    version=VERSION,
    author='Masamichi iizumi',
    author_email='m.iizumi@miosync.email',
    description='GPU-accelerated Lambda³ framework for MD trajectory analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/miosync-masa/bankai',
    project_urls={
        'Bug Reports': 'https://github.com/miosync-masa/bankai/issues',
        'Source': 'https://github.com/miosync-masa/bankai',
        'Documentation': '',
    },
    
    # パッケージ設定
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    package_dir={'bankai': 'bankai'},
    
    # Pythonバージョン
    python_requires='>=3.8',
    
    # 依存関係
    install_requires=install_requires,
    extras_require=extras_require,
    
    # エントリーポイント
    entry_points={
        'console_scripts': [
            'lambda3-benchmark=lambda3_gpu.benchmarks.performance_tests:main',
            'lambda3-quick-test=lambda3_gpu.benchmarks.performance_tests:run_quick_benchmark',
            'lambda3-check-gpu=lambda3_gpu.utils.gpu_check:main',  # 新規追加！
        ],
    },
    
    # 分類
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: CUDA',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Environment :: GPU :: NVIDIA CUDA :: 11.0',
        'Environment :: GPU :: NVIDIA CUDA :: 11.2',
        'Environment :: GPU :: NVIDIA CUDA :: 11.4',
        'Environment :: GPU :: NVIDIA CUDA :: 11.6',
        'Environment :: GPU :: NVIDIA CUDA :: 11.8',
        'Environment :: GPU :: NVIDIA CUDA :: 12.0',
        'Environment :: GPU :: NVIDIA CUDA :: 12.1',
        'Environment :: GPU :: NVIDIA CUDA :: 12.2',
        'Environment :: GPU :: NVIDIA CUDA :: 12.3',
        'Environment :: GPU :: NVIDIA CUDA :: 12.4',
        'Environment :: GPU :: NVIDIA CUDA :: 12.5',
    ],
    
    # キーワード
    keywords='molecular-dynamics gpu cuda lambda3 trajectory-analysis protein-folding structural-analysis',
    
    # データファイル
    include_package_data=True,
    package_data={
        'lambda3_gpu': [
            'README.md',
            'LICENSE',
            'requirements.txt',
            'cuda_compatibility.json',  # CUDA互換性情報
        ],
    },
    
    # テスト
    test_suite='tests',
    tests_require=[
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
    ],
    
    # プラットフォーム
    platforms=['Linux', 'Windows'],
    
    # ライセンス
    license='MIT',
    
    # その他のメタデータ
    zip_safe=False,  # CUDAバイナリのため
)

# インストール後メッセージ（改良版）
def print_post_install_message():
    """インストール後のメッセージを表示"""
    cuda_info = f"CUDA {cuda_version_str}" if cuda_version_str else "No CUDA"
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    Lambda³ GPU Installation                    ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  🎉 Installation complete!                                    ║
║                                                               ║""")
    
    print(f"║  📊 Environment: {cuda_info:<43} ║")
    
    if cuda_major == 12 and cuda_minor >= 5:
        print("║  ⚠️  CUDA 12.5 compatibility mode enabled                    ║")
        print("║     Consider downgrading to CUDA 12.4 for best results      ║")
    
    print("""║                                                               ║
║  Quick test:                                                  ║
║    $ python -c "from lambda3_gpu import MDLambda3DetectorGPU" ║
║    $ lambda3-check-gpu  # Check GPU status                   ║
║    $ lambda3-quick-test # Run benchmark                      ║
║                                                               ║
║  Documentation:                                               ║
║    　　　　　　　　　　　　　　　　　　　　　　                      ║
║                                                               ║
║  Need help?                                                   ║
║    - GitHub Issues: https://github.com/.../issues           ║
║    - Email: info@miosync.email                             ║
║                                                               ║
║  NO TIME, NO PHYSICS, ONLY STRUCTURE! 🌌                     ║
║                          with GPU POWER! ⚡                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

💕 Thank you for using Lambda³ GPU! - by 環ちゃん
""")

# メッセージ表示
print_post_install_message()
