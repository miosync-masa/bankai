"""
Lambda³ Structures Module
~~~~~~~~~~~~~~~~~~~~~~~~~

Lambda³構造計算モジュール 💕

Components:
    - LambdaStructuresCore: 汎用Lambda³構造計算（CPU, domain-agnostic）
    - LambdaStructuresGPU: GPU版Lambda構造計算（MD特化）
    - MDFeaturesGPU: MD特徴抽出
    - TensorOperationsGPU: テンソル演算
"""

# ── Core（汎用版・CPU・ドメイン非依存） ──
from .lambda_structures_core import (
    LambdaCoreConfig,
    LambdaStructuresCore,
)

# ── GPU版（MD特化） ──
from .lambda_structures_gpu import (
    LambdaStructuresGPU,
    compute_adaptive_window_size_gpu,
    compute_coupling_strength_gpu,
    compute_lambda_structures_gpu,
    compute_local_fractal_dimension_gpu,
    compute_structural_coherence_gpu,
    compute_structural_entropy_gpu,
)
from .md_features_gpu import (
    MDFeaturesGPU,
    calculate_contacts_gpu,
    calculate_dihedrals_gpu,
    calculate_radius_of_gyration_gpu,
    calculate_rmsd_gpu,
    extract_md_features_gpu,
)
from .tensor_operations_gpu import (
    TensorOperationsGPU,
    batch_tensor_operation,
    compute_correlation_gpu,
    compute_covariance_gpu,
    compute_gradient_gpu,
    sliding_window_operation_gpu,
)

__all__ = [
    # Lambda Structures Core (domain-agnostic)
    "LambdaStructuresCore",
    "LambdaCoreConfig",
    # Lambda Structures GPU (MD-specific)
    "LambdaStructuresGPU",
    "compute_lambda_structures_gpu",
    "compute_adaptive_window_size_gpu",
    "compute_structural_coherence_gpu",
    "compute_local_fractal_dimension_gpu",
    "compute_coupling_strength_gpu",
    "compute_structural_entropy_gpu",
    # MD Features
    "MDFeaturesGPU",
    "extract_md_features_gpu",
    "calculate_rmsd_gpu",
    "calculate_radius_of_gyration_gpu",
    "calculate_contacts_gpu",
    "calculate_dihedrals_gpu",
    # Tensor Operations
    "TensorOperationsGPU",
    "compute_gradient_gpu",
    "compute_covariance_gpu",
    "compute_correlation_gpu",
    "sliding_window_operation_gpu",
    "batch_tensor_operation",
]

# 初期化メッセージ
import logging

logger = logging.getLogger("bankai.structures")
logger.debug("Lambda³ Structures module initialized (Core + GPU)")
