# BANKAI（Bond-vector ANalysis of Kinetic Amino acid Initiator）

GPU-accelerated sub-picosecond causal cascade detection in GROMACS molecular dynamics trajectories.

---

## Overview

Conventional MD analysis reduces raw atomic coordinates to summary metrics like RMSD or RMSF — discarding the vast majority of structural information before analysis even begins. **BANKAI takes the opposite approach**: it operates directly on the full atomic coordinate trajectory from GROMACS, analyzing every atom at every timestep without dimensionality reduction.

At **0.01 picosecond (10 femtosecond) resolution**, thermal noise dominates the signal by 10:1. BANKAI's **4-layer statistical filtering architecture** strips away thermal fluctuations layer by layer, achieving a final S/N ratio exceeding 100:1 — making it possible to detect statistically significant structural events that are completely invisible to conventional tools.

**Core equation:**

```
ΔΛC = ρT · σS · |ΛF|
```

Where `ΔΛC` is the causal cascade event, `ρT` is tension density, `σS` is structural synchronization, and `ΛF` is the lambda field vector.

## Why BANKAI?

**The problem:** At 0.01 ps resolution, thermal energy at 300K (kT ≈ 2.5 kJ/mol) causes atomic vibrations that completely bury biologically meaningful structural changes. Traditional tools either avoid this timescale entirely, or collapse the data into aggregate metrics (RMSD, radius of gyration) that erase the very signals you need.

**BANKAI's answer:** Don't reduce — resolve. Analyze the full atomic coordinate tensor directly, and use physics-informed statistical filtering to separate signal from noise.

## Key Features

- **Full-atom analysis** — No RMSD reduction. Every atom, every frame, raw coordinates
- **Sub-picosecond resolution** — 0.01 ps intervals, 100,001 frames per nanosecond
- **4-layer noise filtering** — Thermal fluctuation removal achieving S/N > 100:1
- **20+ custom CUDA kernels** — Not just CuPy wrappers; hand-optimized GPU kernels with shared memory, coalesced access, and atomic operations for 150–200x speedup over CPU
- **Automatic CPU fallback** — Works without GPU (slower but fully functional)
- **Two-stage analysis** — Macro-level event detection → Residue-level causal tracing
- **Genesis atom identification** — Pinpoints the first atom to trigger a cascade event
- **Phase space dynamics** — Lyapunov exponents, attractor characterization, recurrence quantification
- **Causal network mapping** — Residue-to-residue causality with confidence scoring
- **Built-in visualization** — Publication-ready plots and interactive 3D networks

## 4-Layer Analysis Architecture

```
Input: 0.01 ps GROMACS trajectory (full atomic coordinates, thermal noise dominant)
  │
  ├─ Layer 1: Λ³ Structural Analysis ──── Multi-scale statistical filtering (~80% noise reduction)
  │   • 3 concurrent timescales (σ₁=short, σ₂=mid, σ₃=long)
  │   • Adaptive 3σ–5σ significance thresholds
  │
  ├─ Layer 2: Topological Break Detection ── Structural continuity monitoring (~60% residual removal)
  │   • Q_lambda (topological charge): winding number of ΛF flow
  │   • Irreversible vs reversible change discrimination
  │
  ├─ Layer 3: 3-Axis Anomaly Scoring ──── Physics-based validation
  │   • Spatial: directional/cooperative vs random/isotropic
  │   • Synchronization: correlated (>0.6) vs uncorrelated (<0.3)
  │   • Temporal: Maxwell-Boltzmann deviation (>3σ)
  │
  └─ Layer 4: Phase Space Attractor Analysis ── Deterministic dynamics extraction
      • Lyapunov exponents, correlation dimension
      • Recurrence Quantification Analysis (RQA)
      • Attractor compactness vs diffusive noise

Output: Statistically significant structural events with confidence scores
        (S/N > 100:1, configurable 95%–99.9% confidence)
```

## Installation

### From PyPI

```bash
pip install bankai-md
```

### From source

```bash
git clone https://github.com/miosync-masa/bankai.git
cd bankai
pip install -e .
```

### With GPU support

```bash
# CUDA 12.x
pip install -e ".[cuda12]"

# CUDA 11.x
pip install -e ".[cuda11]"

# CUDA 12.5+ (compatibility mode)
pip install -e ".[cuda12-compat]"

# Full (CUDA 12 + visualization + dev tools)
pip install -e ".[full]"
```

### Google Colab

```python
# Step 0: Sample data setup
!pip install gdown -q
import os
import gdown

folder_url = 'https://drive.google.com/drive/folders/1AaS6NA8aCUfIrQArltNERNUotW6Pcayq?usp=drive_link'
folder_id = folder_url.split('/')[-1].split('?')[0]
destination_folder = '/content/'
os.makedirs(destination_folder, exist_ok=True)
gdown.download_folder(
    f'https://drive.google.com/drive/folders/{folder_id}',
    output=destination_folder,
    quiet=False,
    use_cookies=False
)

# Step 1: Install CUDA Toolkit
!apt-get install -y cuda-toolkit-12-2

# Step 2: Configure CUDA environment
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.2'
os.environ['PATH'] = '/usr/local/cuda-12.2/bin:' + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

# Step 3: Install GPU backend
!pip install cupy-cuda12x==12.3.0 --no-cache-dir

# Step 4: Install BANKAI
!pip install bankai-md

# Step 5: Run full analysis
import warnings
warnings.filterwarnings('ignore')

from bankai.analysis.run_full_analysis import run_quantum_validation_pipeline

results = run_quantum_validation_pipeline(
    trajectory_path='/content/demo_gromacs/trajectory_stable.npy',
    metadata_path='/content/demo_gromacs/metadata_stable.json',
    protein_indices_path='/content/demo_gromacs/protein_stable.npy',
    topology_path=None,
    enable_two_stage=True,
    enable_third_impact=True,
    enable_visualization=True,
    output_dir='./gromacs_results_v4',
    verbose=True,
    atom_mapping_path='/content/demo_gromacs/residue_atom_mapping.json',
    third_impact_top_n=10
)
```

> **⚠️ Troubleshooting:** Depending on the Colab runtime version,
> dependency conflicts may occur. If you encounter errors, try:
> ```python
> !pip install xarray==2023.7.0
> !pip install pylibraft-cu12==24.10.0
> ```

### Requirements

- Python 3.9+
- CUDA Compute Capability 7.0+ (V100, A100, H100, RTX series)
- CuPy 12.0+ (matched to your CUDA version)
- NumPy < 2.0.0
- GROMACS trajectory data (.npy format)

## Quick Start

### CLI

```bash
# Show help
bankai --help

# Check GPU environment
bankai info

# Run analysis with sample data
bankai example

# Run analysis on your data
bankai run trajectory.npy --protein-indices protein.npy --metadata metadata.json

# Full two-stage analysis
bankai full trajectory.npy --protein-indices protein.npy \
    --events 5000:10000:unfolding 20000:25000:aggregation \
    --n-residues 129
```

### Python API

```python
import bankai
from bankai import MDLambda3DetectorGPU, MDConfig

# Configure
config = MDConfig()
config.use_extended_detection = True
config.use_phase_space = True

# Initialize detector (auto GPU/CPU selection)
detector = MDLambda3DetectorGPU(config)

# Run analysis
result = detector.analyze(trajectory, backbone_indices)

# Visualize
from bankai.visualization import Lambda3VisualizerGPU
visualizer = Lambda3VisualizerGPU()
fig = visualizer.visualize_results(result)
```

### Two-Stage Analysis (Residue-Level Causality)

```python
from bankai import TwoStageAnalyzerGPU, perform_two_stage_analysis_gpu

events = [
    (5000, 10000, 'unfolding'),
    (20000, 25000, 'aggregation')
]

two_stage_result = perform_two_stage_analysis_gpu(
    trajectory, macro_result, events, n_residues=129
)

# Causal network visualization
from bankai.visualization import CausalityVisualizerGPU
viz = CausalityVisualizerGPU()
fig = viz.visualize_residue_causality(
    two_stage_result.residue_analyses['unfolding'],
    interactive=True
)
```

## Architecture

```
bankai/
├── __init__.py          # Public API, GPU detection, lazy imports
├── __main__.py          # python -m bankai entrypoint
├── cli.py               # CLI (bankai command)
├── models.py            # Result types & data models
├── core/                # GPU kernels, memory management, utilities
│   ├── gpu_kernels.py       # Low-level CUDA kernel wrappers
│   ├── gpu_memory.py        # GPU memory pool & batch management
│   ├── gpu_utils.py         # Array operations, CPU/GPU dispatch
│   └── gpu_patches.py       # CuPy compatibility patches
├── analysis/            # Main analysis engines
│   ├── md_lambda3_detector_gpu.py   # Core Λ³ detector
│   ├── two_stage_analyzer_gpu.py    # Two-stage (macro→residue) pipeline
│   ├── run_full_analysis.py        # End-to-end pipeline orchestrator
│   ├── third_impact_analytics.py   # Advanced cascade analytics
│   └── maximum_report_generator.py # Comprehensive report generation
├── detection/           # Anomaly & event detection
│   ├── anomaly_detection_gpu.py    # Statistical anomaly detection
│   ├── boundary_detection_gpu.py   # Phase boundary identification
│   ├── extended_detection_gpu.py   # Extended event detection
│   ├── phase_space_gpu.py          # Phase space reconstruction & analysis
│   └── topology_breaks_gpu.py      # Topological break detection
├── residue/             # Residue-level analysis
│   ├── causality_analysis_gpu.py   # Inter-residue causal inference
│   ├── confidence_analysis_gpu.py  # Statistical confidence scoring
│   ├── residue_network_gpu.py      # Residue interaction networks
│   └── residue_structures_gpu.py   # Structural feature extraction
├── structures/          # Structural computation
│   ├── lambda_structures_gpu.py    # Λ-structure tensor computation
│   ├── md_features_gpu.py          # MD feature extraction
│   └── tensor_operations_gpu.py    # Core tensor math
├── quantum/             # Quantum-level validation
│   └── quantum_validation_gpu.py   # Quantum effect detection
├── visualization/       # Plotting & interactive viz
│   ├── plot_results_gpu.py         # Static plots (matplotlib)
│   └── causality_viz_gpu.py        # Causal network viz (plotly)
├── benchmark/           # Performance testing
│   └── performance_tests.py
└── data/                # Sample datasets
    └── chignolin/           # Chignolin mini-protein test data
```

## Performance

### End-to-End Pipeline

| Data Size    | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 1K frames   | ~10s     | ~0.5s    | 20x     |
| 10K frames  | ~120s    | ~5s      | 24x     |
| 50K frames  | ~800s    | ~25s     | 32x     |
| 100K frames | ~2000s   | ~50s     | 40x     |

### Custom CUDA Kernels (100 atoms × 5,000 frames)

| Kernel | CPU | CuPy (generic) | BANKAI Kernel | Speedup |
|--------|-----|-----------------|---------------|---------|
| Tension field (ρT) | 1200s | 120s | 8s | **150x** |
| Topological charge (Q_λ) | 800s | 80s | 4s | **200x** |
| Anomaly detection | 600s | 60s | 3s | **200x** |
| Phase space analysis | 2400s | 240s | 15s | **160x** |

BANKAI's custom kernels are not CuPy wrappers — they are hand-written CUDA with shared memory tiling, coalesced access patterns, and lock-free atomic reductions. This is what makes 0.01 ps analysis feasible within hours rather than weeks.

*Benchmarked on NVIDIA RTX 4070 Ti SUPER*

## Configuration

### Environment Variables

| Variable | Description |
|----------|------------|
| `BANKAI_GPU_MEMORY_LIMIT` | GPU memory limit in GB (e.g., `"8.0"`) |
| `BANKAI_DEBUG` | Enable debug logging (`"1"` or `"true"`) |
| `BANKAI_NO_BANNER` | Suppress CLI banner |
| `BANKAI_BANNER_STYLE` | CLI banner style (`simple`, `ascii`, `matrix`) |

### Memory Management

```python
# Set GPU memory limit
import os
os.environ['BANKAI_GPU_MEMORY_LIMIT'] = '8.0'

# Or via detector
detector.memory_manager.set_max_memory(8)
detector.set_batch_size(5000)

# Mixed precision (FP16)
detector.enable_mixed_precision()
```

## Sample Data

BANKAI includes a Chignolin mini-protein dataset for testing:

```python
from bankai.data import load_chignolin, chignolin_available

if chignolin_available():
    data = load_chignolin()
    trajectory = data['trajectory']       # (10001, 166, 3)
    metadata = data['metadata']
    protein_indices = data['protein_indices']
```

Generate synthetic test data:

```python
from bankai.data import generate_synthetic_chignolin
paths = generate_synthetic_chignolin()
```

Or via CLI:

```bash
bankai example --generate
```

## Troubleshooting

**GPU not detected:**

```python
from bankai import get_gpu_info
print(get_gpu_info())
```

**Out of memory:**

Reduce batch size or disable extended features:

```python
config.gpu_batch_size = 1000
config.use_extended_detection = False
config.use_phase_space = False
```

**NumPy 2.0 compatibility:**

BANKAI requires NumPy < 2.0.0. If you see `numpy._core` errors, downgrade:

```bash
pip install "numpy>=1.22.0,<2.0.0"
```

## Pharmaceutical Applications

BANKAI enables atomic-level analysis previously inaccessible to conventional MD tools:

- **Drug-protein interactions** — Visualize binding processes at atomic resolution, including transient hydrogen bond formation (10–50 fs) and proton transfer events (20–100 fs)
- **Allosteric pathway mapping** — Trace how structural perturbations propagate across residue networks with causal directionality
- **Cryptic binding site discovery** — Detect transient pocket openings invisible to ensemble-averaged structures
- **Resistance mutation analysis** — Identify how point mutations alter cascade propagation pathways
- **QM/MM candidate screening** — Efficiently identify statistically anomalous events (>5σ) that warrant quantum-mechanical investigation

## Author

**Masamichi Iizumi** — CEO, Miosync, Inc.

- GitHub: [miosync-masa](https://github.com/miosync-masa)
- Email: m.iizumi@miosync.email

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with 💕 by Masamichi & Tamaki*
