"""
Lambda³ Structure Computation (GPU Version) — v1.2.1 NaN-guard patch
=====================================================================

Differences from v1.2.0:
- Input validation (com_positions, rmsd, radius_of_gyration finiteness)
- _compute_rho_T: scalar covariance guard (ported from lambda_structures_core.py)
- _compute_sigma_s: NaN guard on correlation matrix
- _compute_lambda_F: post-computation finiteness check (warning)
- _print_statistics: NaN/Inf prominently flagged

API, return-value keys, and array shapes are UNCHANGED.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
    print("✅ CuPy successfully imported")
except ImportError as e:
    cp = None
    HAS_GPU = False
    print(f"⚠️ CuPy not available: {e}")

from ..core.gpu_memory import GPUMemoryManager
from ..core.gpu_utils import GPUBackend
from ..models import NDArray

tension_field_kernel = None
topological_charge_kernel = None

if HAS_GPU:
    try:
        from ..core import tension_field_kernel, topological_charge_kernel
        print("✅ Custom kernels imported")
    except ImportError as e:
        print(f"⚠️ Custom kernels not available: {e}")

logger = logging.getLogger(__name__)


@dataclass
class LambdaStructureConfig:
    use_mixed_precision: bool = True
    batch_size: Optional[int] = None
    cache_intermediates: bool = True
    profile: bool = False


class LambdaStructuresGPU(GPUBackend):
    """
    Lambda³ structure computation (GPU implementation).
    v1.2.1: NaN-guarded throughout the pipeline.
    """

    def __init__(
        self,
        config: Optional[LambdaStructureConfig] = None,
        memory_manager: Optional[GPUMemoryManager] = None,
        force_cpu: bool = False,
        device: Union[str, int] = "auto",
        **kwargs,
    ):
        self.config = config or LambdaStructureConfig()
        super().__init__(
            device=device,
            force_cpu=force_cpu,
            mixed_precision=self.config.use_mixed_precision,
            profile=self.config.profile,
            memory_manager_config=None,
            **kwargs,
        )
        self._cache = {} if self.config.cache_intermediates else None
        if memory_manager is not None:
            self.memory_manager = memory_manager

        logger.info("✅ LambdaStructuresGPU initialized:")
        logger.info(f"   Backend: {self.xp.__name__}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   GPU Mode: {self.is_gpu}")
        self._verify_backend()

    def _verify_backend(self):
        try:
            if not hasattr(self, "xp") or self.xp is None:
                raise RuntimeError("xp not initialized by parent class!")
            test = self.xp.array([1, 2, 3])
            diff = self.xp.diff(test)
            norm = self.xp.linalg.norm(diff)
            logger.info(f"✅ Backend test passed: diff={diff}, norm={norm}")
        except Exception as e:
            logger.error(f"❌ Backend verification failed: {e}")
            raise

    # ================================================================
    # ★ NEW: Input validation helper
    # ================================================================
    def _validate_input_array(self, arr: np.ndarray, name: str) -> None:
        """
        Verify that an input array contains no NaN/Inf values.

        Raises ValueError with diagnostic info on the first 5 bad frames
        so upstream pipeline issues (e.g. broken residue mapping) can be
        traced quickly instead of producing a silent NaN output.
        """
        if arr is None:
            return
        arr_np = np.asarray(arr)
        if arr_np.size == 0:
            return
        if not np.all(np.isfinite(arr_np)):
            n_bad = int(np.sum(~np.isfinite(arr_np)))
            n_total = int(arr_np.size)
            # 1D の場合は単純に bad index, ND の場合はフレーム単位で探す
            if arr_np.ndim == 1:
                bad_idx = np.where(~np.isfinite(arr_np))[0]
            else:
                # フレーム軸 (axis=0) で見て、どこかに NaN/Inf があるフレームを抽出
                bad_mask = ~np.isfinite(arr_np).reshape(arr_np.shape[0], -1).all(axis=1)
                bad_idx = np.where(bad_mask)[0]
            raise ValueError(
                f"md_features['{name}'] contains {n_bad}/{n_total} non-finite values. "
                f"First bad frames/indices: {bad_idx[:5].tolist()}. "
                f"This typically indicates an upstream issue such as a residue "
                f"mapping mismatch (e.g. atom_mapping.json built for a different "
                f"system than the trajectory)."
            )

    # ================================================================
    # Main entrypoint
    # ================================================================
    def compute_lambda_structures(
        self,
        trajectory: np.ndarray,
        md_features: dict[str, np.ndarray],
        window_steps: int,
    ) -> dict[str, np.ndarray]:
        try:
            with self.timer("compute_lambda_structures"):
                logger.info(
                    f"🚀 Computing Lambda³ structures "
                    f"(window={window_steps}, mode={'GPU' if self.is_gpu else 'CPU'})"
                )

                # ★ NEW: 入力検証（fail-fast）
                if "com_positions" not in md_features:
                    raise ValueError("com_positions not found in md_features")
                self._validate_input_array(md_features["com_positions"], "com_positions")
                # rmsd / Rg は optional だが、あれば検証
                for key in ("rmsd", "radius_of_gyration"):
                    if key in md_features:
                        self._validate_input_array(md_features[key], key)

                positions_gpu = self.to_gpu(md_features["com_positions"])
                n_frames = positions_gpu.shape[0]
                logger.debug(f"Processing {n_frames} frames")

                with self.timer("lambda_F"):
                    lambda_F, lambda_F_mag = self._compute_lambda_F(positions_gpu)
                with self.timer("lambda_FF"):
                    lambda_FF, lambda_FF_mag = self._compute_lambda_FF(lambda_F)
                with self.timer("rho_T"):
                    rho_T = self._compute_rho_T(positions_gpu, window_steps)
                with self.timer("Q_lambda"):
                    Q_lambda, Q_cumulative = self._compute_Q_lambda(lambda_F, lambda_F_mag)
                with self.timer("sigma_s"):
                    sigma_s = self._compute_sigma_s(md_features, window_steps)
                with self.timer("coherence"):
                    coherence = self._compute_coherence(lambda_F, window_steps)

                results = {
                    "lambda_F": self.to_cpu(lambda_F),
                    "lambda_F_mag": self.to_cpu(lambda_F_mag),
                    "lambda_FF": self.to_cpu(lambda_FF),
                    "lambda_FF_mag": self.to_cpu(lambda_FF_mag),
                    "rho_T": self.to_cpu(rho_T),
                    "Q_lambda": self.to_cpu(Q_lambda),
                    "Q_cumulative": self.to_cpu(Q_cumulative),
                    "sigma_s": self.to_cpu(sigma_s),
                    "structural_coherence": self.to_cpu(coherence),
                }
                self._print_statistics(results)
                return results

        except Exception as e:
            logger.error(f"❌ Error in compute_lambda_structures: {e}")
            logger.error(f"   xp={self.xp if hasattr(self, 'xp') else 'NOT SET'}")
            logger.error(f"   is_gpu={self.is_gpu if hasattr(self, 'is_gpu') else 'NOT SET'}")
            if self.is_gpu and "out of memory" in str(e).lower():
                logger.info("💡 Try reducing batch_size or use force_cpu=True")
            raise

    # ================================================================
    # ΛF: post-computation finiteness check
    # ================================================================
    def _compute_lambda_F(self, positions: NDArray) -> tuple[NDArray, NDArray]:
        lambda_F = self.xp.diff(positions, axis=0)
        lambda_F_mag = self.xp.linalg.norm(lambda_F, axis=1)

        # ★ NEW: 出力の健全性チェック（warning だけ、raise はしない）
        try:
            mag_cpu = self.to_cpu(lambda_F_mag)
            n_bad = int(np.sum(~np.isfinite(mag_cpu)))
            if n_bad > 0:
                logger.warning(
                    f"⚠️ Lambda_F_mag contains {n_bad}/{len(mag_cpu)} non-finite values. "
                    f"Downstream metrics may be invalid; check upstream positions."
                )
        except Exception:
            pass

        return lambda_F, lambda_F_mag

    def _compute_lambda_FF(self, lambda_F: NDArray) -> tuple[NDArray, NDArray]:
        lambda_FF = self.xp.diff(lambda_F, axis=0)
        lambda_FF_mag = self.xp.linalg.norm(lambda_FF, axis=1)
        return lambda_FF, lambda_FF_mag

    # ================================================================
    # ρT: scalar covariance guard (core版から移植)
    # ================================================================
    def _compute_rho_T(self, positions: NDArray, window_steps: int) -> NDArray:
        n_frames = len(positions)

        if self.is_gpu and tension_field_kernel is not None:
            try:
                return tension_field_kernel(positions, window_steps)
            except Exception as e:
                logger.warning(f"Custom kernel failed: {e}, using fallback")

        rho_T = self.xp.zeros(n_frames)
        for step in range(n_frames):
            start = max(0, step - window_steps)
            end = min(n_frames, step + window_steps + 1)
            local_positions = positions[start:end]
            if len(local_positions) > 1:
                centered = local_positions - self.xp.mean(local_positions, axis=0, keepdims=True)
                cov = self.xp.cov(centered.T)
                # ★ NEW: scalar guard (core版思想)
                if cov.ndim == 0:
                    val = float(cov)
                    rho_T[step] = val if np.isfinite(val) else 0.0
                else:
                    tr = self.xp.trace(cov)
                    # ★ NEW: NaN ガード
                    tr_cpu = float(self.to_cpu(tr)) if hasattr(self, "to_cpu") else float(tr)
                    rho_T[step] = tr if np.isfinite(tr_cpu) else 0.0
        return rho_T

    def _compute_Q_lambda(
        self, lambda_F: NDArray, lambda_F_mag: NDArray
    ) -> tuple[NDArray, NDArray]:
        n_steps = len(lambda_F_mag)

        if self.is_gpu and topological_charge_kernel is not None:
            try:
                Q_lambda = topological_charge_kernel(lambda_F, lambda_F_mag)
                Q_cumulative = self.xp.cumsum(Q_lambda)
                return Q_lambda, Q_cumulative
            except Exception as e:
                logger.warning(f"Custom kernel failed: {e}, using fallback")

        Q_lambda = self.xp.zeros(n_steps)
        for step in range(1, n_steps):
            if lambda_F_mag[step] > 1e-10 and lambda_F_mag[step - 1] > 1e-10:
                v1 = lambda_F[step - 1] / lambda_F_mag[step - 1]
                v2 = lambda_F[step] / lambda_F_mag[step]
                cos_angle = self.xp.clip(self.xp.dot(v1, v2), -1, 1)
                angle = self.xp.arccos(cos_angle)
                if len(v1) >= 2:
                    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
                    signed_angle = angle if cross_z >= 0 else -angle
                else:
                    signed_angle = angle
                Q_lambda[step] = signed_angle / (2 * self.xp.pi)

        Q_cumulative = self.xp.cumsum(Q_lambda)
        return Q_lambda, Q_cumulative

    # ================================================================
    # σ_s: NaN guard on correlation matrix
    # ================================================================
    def _compute_sigma_s(self, md_features: dict[str, np.ndarray], window_steps: int) -> NDArray:
        if "rmsd" not in md_features or "radius_of_gyration" not in md_features:
            n_frames = len(md_features.get("com_positions", []))
            return self.xp.zeros(n_frames)

        rmsd = self.to_gpu(md_features["rmsd"])
        rg = self.to_gpu(md_features["radius_of_gyration"])
        n_frames = len(rmsd)
        sigma_s = self.xp.zeros(n_frames)

        for step in range(n_frames):
            start = max(0, step - window_steps)
            end = min(n_frames, step + window_steps + 1)
            if end - start > 2:
                local_rmsd = rmsd[start:end]
                local_rg = rg[start:end]
                std_rmsd = self.xp.std(local_rmsd)
                std_rg = self.xp.std(local_rg)
                if std_rmsd > 1e-10 and std_rg > 1e-10:
                    corr_matrix = self.xp.corrcoef(self.xp.stack([local_rmsd, local_rg]))
                    val = self.xp.abs(corr_matrix[0, 1])
                    # ★ NEW: NaN guard (core版思想)
                    val_cpu = float(self.to_cpu(val)) if hasattr(self, "to_cpu") else float(val)
                    if np.isfinite(val_cpu):
                        sigma_s[step] = val
                    # NaN なら 0 のまま（zeros 初期化済み）
        return sigma_s

    def _compute_coherence(self, lambda_F: NDArray, window: int) -> NDArray:
        n_frames = len(lambda_F)
        coherence = self.xp.zeros(n_frames)
        for i in range(window, n_frames - window):
            local_F = lambda_F[i - window : i + window]
            mean_dir = self.xp.mean(local_F, axis=0, keepdims=True).ravel()
            mean_norm = self.xp.linalg.norm(mean_dir)
            if mean_norm > 1e-10:
                mean_dir /= mean_norm
                norms = self.xp.linalg.norm(local_F, axis=1)
                valid_mask = norms > 1e-10
                if self.xp.any(valid_mask):
                    normalized_F = local_F[valid_mask] / norms[valid_mask, self.xp.newaxis]
                    coherences = self.xp.dot(normalized_F, mean_dir)
                    coherence[i] = self.xp.mean(coherences)
        return coherence

    # ================================================================
    # ★ NEW: Statistics with prominent NaN/Inf flagging
    # ================================================================
    def _print_statistics(self, results: dict[str, np.ndarray]):
        try:
            logger.info("📊 Lambda Structure Statistics:")
            for key in [
                "lambda_F_mag",
                "lambda_FF_mag",
                "rho_T",
                "Q_cumulative",
                "sigma_s",
            ]:
                if key in results and len(results[key]) > 0:
                    data = np.asarray(results[key])
                    n_total = data.size
                    finite_mask = np.isfinite(data)
                    n_finite = int(np.sum(finite_mask))

                    if n_finite == 0:
                        logger.warning(
                            f"   ⚠️ {key}: ALL {n_total} VALUES NON-FINITE — "
                            f"pipeline output is invalid for this metric."
                        )
                    elif n_finite < n_total:
                        n_bad = n_total - n_finite
                        finite_data = data[finite_mask]
                        logger.warning(
                            f"   ⚠️ {key}: {n_bad}/{n_total} non-finite. "
                            f"Finite stats: min={np.min(finite_data):.3e}, "
                            f"max={np.max(finite_data):.3e}, mean={np.mean(finite_data):.3e}"
                        )
                    else:
                        logger.info(
                            f"   {key}: min={np.min(data):.3e}, max={np.max(data):.3e}, "
                            f"mean={np.mean(data):.3e}, std={np.std(data):.3e}"
                        )
        except Exception as e:
            logger.warning(f"Failed to print statistics: {e}")

    # ================================================================
    # Adaptive window — unchanged
    # ================================================================
    def compute_adaptive_window_size(
        self,
        md_features: dict[str, np.ndarray],
        lambda_structures: dict[str, np.ndarray],
        n_frames: int,
        config: any,
    ) -> dict[str, Union[int, float, dict]]:
        try:
            base_window = int(n_frames * config.window_scale)
            volatility_metrics = {}

            if "rmsd" in md_features:
                rmsd = self.to_gpu(md_features["rmsd"])
                mean_val = float(self.xp.mean(rmsd))
                volatility_metrics["rmsd"] = (
                    float(self.xp.std(rmsd) / mean_val) if abs(mean_val) > 1e-10 else 0.0
                )

            if "lambda_F_mag" in lambda_structures:
                lf_mag = self.to_gpu(lambda_structures["lambda_F_mag"])
                mean_val = float(self.xp.mean(lf_mag))
                volatility_metrics["lambda_f"] = (
                    float(self.xp.std(lf_mag) / mean_val) if abs(mean_val) > 1e-10 else 0.0
                )

            if "rho_T" in lambda_structures:
                rho_t = self.to_gpu(lambda_structures["rho_T"])
                mean_val = float(self.xp.mean(rho_t))
                volatility_metrics["rho_t"] = (
                    float(self.xp.std(rho_t) / mean_val) if abs(mean_val) > 1e-10 else 0.0
                )

            scale_factor = 1.0
            if volatility_metrics.get("rmsd", 0) > 0.5:
                scale_factor *= 0.7
            elif volatility_metrics.get("rmsd", 0) < 0.1:
                scale_factor *= 1.5
            if volatility_metrics.get("lambda_f", 0) > 1.0:
                scale_factor *= 0.8
            elif volatility_metrics.get("lambda_f", 0) < 0.2:
                scale_factor *= 1.3

            adaptive_window = int(base_window * scale_factor)
            adaptive_window = np.clip(adaptive_window, config.min_window, config.max_window)

            windows = {
                "primary": adaptive_window,
                "fast": max(config.min_window, adaptive_window // 2),
                "slow": min(config.max_window, adaptive_window * 2),
                "boundary": max(10, adaptive_window // 3),
                "scale_factor": scale_factor,
                "volatility_metrics": volatility_metrics,
            }
            logger.info(
                f"🎯 Adaptive window sizes: primary={windows['primary']}, "
                f"scale_factor={scale_factor:.2f}"
            )
            return windows

        except Exception as e:
            logger.error(f"Failed to compute adaptive window size: {e}")
            return {
                "primary": 50, "fast": 25, "slow": 100, "boundary": 10,
                "scale_factor": 1.0, "volatility_metrics": {},
            }


# ===============================
# Standalone helpers (unchanged)
# ===============================
def compute_lambda_structures_gpu(
    trajectory: np.ndarray,
    md_features: dict[str, np.ndarray],
    window_steps: int,
    config: Optional[LambdaStructureConfig] = None,
    memory_manager: Optional[GPUMemoryManager] = None,
    force_cpu: bool = False,
) -> dict[str, np.ndarray]:
    calculator = LambdaStructuresGPU(config, memory_manager, force_cpu=force_cpu)
    return calculator.compute_lambda_structures(trajectory, md_features, window_steps)


def compute_adaptive_window_size_gpu(
    md_features: dict[str, np.ndarray],
    lambda_structures: dict[str, np.ndarray],
    n_frames: int,
    config: any,
    force_cpu: bool = False,
) -> dict[str, Union[int, float, dict]]:
    calculator = LambdaStructuresGPU(force_cpu=force_cpu)
    return calculator.compute_adaptive_window_size(md_features, lambda_structures, n_frames, config)


def compute_structural_coherence_gpu(
    lambda_F: np.ndarray, window: int = 50, force_cpu: bool = False
) -> np.ndarray:
    calculator = LambdaStructuresGPU(force_cpu=force_cpu)
    lambda_F_gpu = calculator.to_gpu(lambda_F)
    coherence = calculator._compute_coherence(lambda_F_gpu, window)
    return calculator.to_cpu(coherence)


def compute_local_fractal_dimension_gpu(
    q_cumulative: np.ndarray, window: int = 50, force_cpu: bool = False
) -> np.ndarray:
    from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
    detector = BoundaryDetectorGPU(force_cpu=force_cpu)
    return detector._compute_fractal_dimensions_gpu(q_cumulative, window)


def compute_coupling_strength_gpu(
    q_cumulative: np.ndarray, window: int = 50, force_cpu: bool = False
) -> np.ndarray:
    from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
    detector = BoundaryDetectorGPU(force_cpu=force_cpu)
    return detector._compute_coupling_strength_gpu(q_cumulative, window)


def compute_structural_entropy_gpu(
    rho_t: np.ndarray, window: int = 50, force_cpu: bool = False
) -> np.ndarray:
    from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
    detector = BoundaryDetectorGPU(force_cpu=force_cpu)
    return detector._compute_structural_entropy_gpu(rho_t, window)


# ===============================
# エクスポート
# ===============================
__all__ = [
    "LambdaStructuresGPU",
    "LambdaStructureConfig",
    "compute_lambda_structures_gpu",
    "compute_adaptive_window_size_gpu",
    "compute_structural_coherence_gpu",
    "compute_local_fractal_dimension_gpu",
    "compute_coupling_strength_gpu",
    "compute_structural_entropy_gpu",
]
