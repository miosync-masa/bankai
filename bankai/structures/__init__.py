"""
Lambda³ GPU Structures Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lambda³構造計算のGPU実装だよ〜！💕
ΛF, ΛFF, ρT, Q_Λ, σₛを超高速で計算しちゃう！

Components:
    - LambdaStructuresGPU: Lambda構造計算
    - MDFeaturesGPU: MD特徴抽出
    - TensorOperationsGPU: テンソル演算
"""

from .lambda_structures_gpu import (
    LambdaStructuresGPU,
    compute_lambda_structures_gpu,
    compute_adaptive_window_size_gpu,
    compute_structural_coherence_gpu,
    compute_local_fractal_dimension_gpu,
    compute_coupling_strength_gpu,
    compute_structural_entropy_gpu,
)

from .md_features_gpu import (
    MDFeaturesGPU,
    extract_md_features_gpu,
    calculate_rmsd_gpu,
    calculate_radius_of_gyration_gpu,
    calculate_contacts_gpu,
    calculate_dihedrals_gpu,
)

from .tensor_operations_gpu import (
    TensorOperationsGPU,
    compute_gradient_gpu,
    compute_covariance_gpu,
    compute_correlation_gpu,
    sliding_window_operation_gpu,
    batch_tensor_operation,
)

__all__ = [
    # Lambda Structures
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

logger = logging.getLogger("lambda3_gpu.structures")
logger.debug("Lambda³ GPU Structures module initialized")
