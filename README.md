BANKAI: A Sub-Picosecond Cascade Origin Identifier for GROMACS Trajectories

MD軌道解析を10-50倍高速化！

## 🌟 特徴

- **完全GPU化**: すべての計算をGPU上で実行
- **後方互換性**: 既存のAPIをそのまま使用可能
- **自動フォールバック**: GPU不在時は自動的にCPUモードで動作
- **メモリ効率**: 大規模データに対応するバッチ処理
- **並列処理**: マルチイベント・マルチ残基の並列解析

## 📦 インストール

```bash
# 依存関係のインストール
pip install -r requirements.txt

# Lambda³ GPUのインストール
python setup.py install
```

### 必要な環境

- Python 3.8+
- CUDA 11.0+
- CuPy 10.0+
- NumPy, SciPy, scikit-learn
- matplotlib, plotly (可視化用)

## 🚀 クイックスタート

### 基本的な使い方

```python
from lambda3_gpu import MDLambda3DetectorGPU, MDConfig

# 設定
config = MDConfig()
config.use_extended_detection = True  # 拡張検出ON
config.use_phase_space = True        # 位相空間解析ON

# GPU検出器の初期化（自動でGPU/CPU選択）
detector = MDLambda3DetectorGPU(config)

# 解析実行
result = detector.analyze(trajectory, backbone_indices)

# 結果の可視化
from lambda3_gpu.visualization import Lambda3VisualizerGPU
visualizer = Lambda3VisualizerGPU()
fig = visualizer.visualize_results(result)
```

### 2段階解析（残基レベル）

```python
from lambda3_gpu import TwoStageAnalyzerGPU, perform_two_stage_analysis_gpu

# イベント定義
events = [
    (5000, 10000, 'unfolding'),
    (20000, 25000, 'aggregation')
]

# 2段階解析
two_stage_result = perform_two_stage_analysis_gpu(
    trajectory, 
    macro_result,
    events,
    n_residues=129
)

# 因果ネットワーク可視化
from lambda3_gpu.visualization import CausalityVisualizerGPU
viz = CausalityVisualizerGPU()
fig = viz.visualize_residue_causality(
    two_stage_result.residue_analyses['unfolding'],
    interactive=True  # インタラクティブ版
)
```

## ⚡ パフォーマンス最適化

### メモリ管理

```python
# メモリ上限設定（16GB）
detector.memory_manager.set_max_memory(16)

# バッチサイズ調整
detector.set_batch_size(5000)  # フレーム数

# 混合精度モード（FP16）
detector.enable_mixed_precision()
```

### ベンチマーク

```python
from lambda3_gpu.benchmarks import run_quick_benchmark

# クイックベンチマーク
run_quick_benchmark()

# 詳細ベンチマーク
from lambda3_gpu.benchmarks import Lambda3BenchmarkSuite
suite = Lambda3BenchmarkSuite()
suite.run_all_benchmarks()
```

## 📊 期待される性能

| データサイズ | CPU時間 | GPU時間 | スピードアップ |
|------------|---------|---------|---------------|
| 1K frames  | 10s     | 0.5s    | 20x           |
| 10K frames | 120s    | 5s      | 24x           |
| 50K frames | 800s    | 25s     | 32x           |
| 100K frames| 2000s   | 50s     | 40x           |

*環境: NVIDIA RTX 3090, Intel i9-10900K

## 🔧 高度な使い方

### カスタム設定

```python
# 詳細な設定
config = MDConfig(
    # Lambda³パラメータ
    window_scale=0.005,
    adaptive_window=True,
    
    # 検出設定
    use_periodic=True,
    use_gradual=True,
    use_drift=True,
    
    # GPU設定
    gpu_batch_size=10000,
    mixed_precision=True
)
```

### 位相空間解析

```python
# 位相空間の詳細解析
if result.phase_space_analysis:
    attractor = result.phase_space_analysis['attractor_features']
    print(f"Lyapunov exponent: {attractor['lyapunov_exponent']}")
    print(f"Correlation dimension: {attractor['correlation_dimension']}")
```

### カスタム可視化

```python
# アニメーション作成
anim = visualizer.create_animation(
    result,
    feature='anomaly',  # or 'lambda_f', 'tension'
    interval=50,        # ms
    save_path='animation.mp4'
)

# インタラクティブ3Dネットワーク
fig = viz.visualize_residue_causality(
    analysis,
    interactive=True,
    save_path='network.html'
)
```

## 🐛 トラブルシューティング

### GPU関連

```python
# GPU情報確認
import cupy as cp
print(cp.cuda.runtime.getDeviceProperties(0))

# メモリクリア
detector.memory_manager.clear_cache()
cp.get_default_memory_pool().free_all_blocks()
```

### メモリ不足

```python
# より小さいバッチサイズ
config.gpu_batch_size = 1000

# 拡張検出を無効化
config.use_extended_detection = False
config.use_phase_space = False
```

**NO TIME, NO PHYSICS, ONLY STRUCTURE!** 🌌
