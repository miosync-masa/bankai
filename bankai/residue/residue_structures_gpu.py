"""
Residue-Level Lambda³ Structure Computation (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

残基レベルのLambda³構造をGPUで計算！
単一フレームイベント（量子もつれ）も正しく処理！💫

by 環ちゃん - 完全修正版 v2
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

# GPU imports
try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# Local imports
from ..core import GPUBackend, GPUMemoryManager, handle_gpu_errors

# residue_com_kernelのインポート（エラー時はNone）
try:
    from ..core import residue_com_kernel
except ImportError:
    residue_com_kernel = None

logger = logging.getLogger("bankai.residue.structures")

# ===============================
# Helper Functions
# ===============================


def get_optimal_block_size(n_elements: int, max_block_size: int = 1024) -> int:
    """最適なブロックサイズを計算"""
    if n_elements <= 32:
        return 32
    elif n_elements <= 64:
        return 64
    elif n_elements <= 128:
        return 128
    elif n_elements <= 256:
        return 256
    elif n_elements <= 512:
        return 512
    else:
        return min(1024, max_block_size)


# ===============================
# Data Classes
# ===============================


@dataclass
class ResidueStructureResult:
    """残基レベル構造計算の結果（形状統一版）"""

    residue_lambda_f: np.ndarray  # (n_frames, n_residues, 3) - パディング済み
    residue_lambda_f_mag: np.ndarray  # (n_frames, n_residues) - パディング済み
    residue_rho_t: np.ndarray  # (n_frames, n_residues)
    residue_coupling: np.ndarray  # (n_frames, n_residues, n_residues)
    residue_coms: np.ndarray  # (n_frames, n_residues, 3)

    @property
    def n_frames(self) -> int:
        """全配列で統一されたフレーム数"""
        return self.residue_coms.shape[0]

    @property
    def n_residues(self) -> int:
        return self.residue_coms.shape[1]

    def validate_shapes(self) -> bool:
        """形状の整合性を確認"""
        n_frames = self.n_frames
        n_residues = self.n_residues

        expected_shapes = {
            "residue_lambda_f": (n_frames, n_residues, 3),
            "residue_lambda_f_mag": (n_frames, n_residues),
            "residue_rho_t": (n_frames, n_residues),
            "residue_coupling": (n_frames, n_residues, n_residues),
            "residue_coms": (n_frames, n_residues, 3),
        }

        for name, expected_shape in expected_shapes.items():
            actual_shape = getattr(self, name).shape
            if actual_shape != expected_shape:
                logger.warning(
                    f"Shape mismatch in {name}: "
                    f"expected {expected_shape}, got {actual_shape}"
                )
                return False

        return True

    def get_summary_stats(self) -> dict[str, float]:
        """統計サマリーを辞書で返す"""
        return {
            "mean_lambda_f": float(np.nanmean(self.residue_lambda_f_mag)),
            "max_lambda_f": float(np.nanmax(self.residue_lambda_f_mag)),
            "mean_rho_t": float(np.nanmean(self.residue_rho_t)),
            "mean_coupling": float(np.nanmean(self.residue_coupling)),
            "n_frames": self.n_frames,
            "n_residues": self.n_residues,
        }


# ===============================
# CUDA Kernels（変更なし）
# ===============================

RESIDUE_TENSION_KERNEL = r"""
extern "C" __global__
void compute_residue_tension_kernel(
    const float* __restrict__ residue_coms,  // (n_frames, n_residues, 3)
    float* __restrict__ rho_t,               // (n_frames, n_residues)
    const int n_frames,
    const int n_residues,
    const int window_size
) {
    const int res_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int frame_id = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (res_id >= n_residues || frame_id >= n_frames) return;
    
    const int window_half = window_size / 2;
    const int start_frame = max(0, frame_id - window_half);
    const int end_frame = min(n_frames, frame_id + window_half + 1);
    const int local_window = end_frame - start_frame;
    
    // 局所平均
    float mean_x = 0.0f, mean_y = 0.0f, mean_z = 0.0f;
    
    for (int f = start_frame; f < end_frame; f++) {
        const int idx = (f * n_residues + res_id) * 3;
        mean_x += residue_coms[idx + 0];
        mean_y += residue_coms[idx + 1];
        mean_z += residue_coms[idx + 2];
    }
    
    mean_x /= local_window;
    mean_y /= local_window;
    mean_z /= local_window;
    
    // 局所分散（トレース）
    float var_sum = 0.0f;
    
    for (int f = start_frame; f < end_frame; f++) {
        const int idx = (f * n_residues + res_id) * 3;
        const float dx = residue_coms[idx + 0] - mean_x;
        const float dy = residue_coms[idx + 1] - mean_y;
        const float dz = residue_coms[idx + 2] - mean_z;
        
        var_sum += dx * dx + dy * dy + dz * dz;
    }
    
    rho_t[frame_id * n_residues + res_id] = var_sum / local_window;
}
"""

RESIDUE_COUPLING_KERNEL = r"""
extern "C" __global__
void compute_residue_coupling_kernel(
    const float* __restrict__ residue_coms,  // (n_residues, 3)
    float* __restrict__ coupling,            // (n_residues, n_residues)
    const int n_residues
) {
    extern __shared__ float tile[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int TILE_DIM = blockDim.x;
    
    const int row = by * TILE_DIM + ty;
    const int col = bx * TILE_DIM + tx;
    
    float sum = 0.0f;
    
    // タイルごとに処理
    for (int phase = 0; phase < 3; phase++) {
        // 共有メモリにロード
        if (row < n_residues && tx < n_residues) {
            tile[ty * TILE_DIM + tx] = residue_coms[row * 3 + phase];
        }
        if (col < n_residues && ty < n_residues) {
            tile[(ty + TILE_DIM) * TILE_DIM + tx] = residue_coms[col * 3 + phase];
        }
        __syncthreads();
        
        // 距離計算
        if (row < n_residues && col < n_residues) {
            const float diff = tile[ty * TILE_DIM + tx] - tile[(ty + TILE_DIM) * TILE_DIM + tx];
            sum += diff * diff;
        }
        __syncthreads();
    }
    
    // カップリング計算（1/(1+距離)）
    if (row < n_residues && col < n_residues) {
        const float distance = sqrtf(sum);
        coupling[row * n_residues + col] = 1.0f / (1.0f + distance);
    }
}
"""

# ===============================
# Residue Structures GPU Class
# ===============================


class ResidueStructuresGPU(GPUBackend):
    """
    残基レベルLambda³構造計算のGPU実装
    単一フレームイベント対応版！
    """

    def __init__(self, memory_manager: Optional[GPUMemoryManager] = None, **kwargs):
        super().__init__(**kwargs)
        self.memory_manager = memory_manager or GPUMemoryManager()

        # カーネルコンパイル
        if self.is_gpu and HAS_GPU:
            self._compile_kernels()

    def _compile_kernels(self):
        """カスタムカーネルをコンパイル"""
        try:
            self.tension_kernel = cp.RawKernel(
                RESIDUE_TENSION_KERNEL, "compute_residue_tension_kernel"
            )
            self.coupling_kernel = cp.RawKernel(
                RESIDUE_COUPLING_KERNEL, "compute_residue_coupling_kernel"
            )
            logger.debug("Residue structure kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile custom kernels: {e}")
            self.tension_kernel = None
            self.coupling_kernel = None

    @handle_gpu_errors
    def compute_residue_structures(
        self,
        trajectory: np.ndarray,
        start_frame: int,
        end_frame: int,
        residue_atoms: dict[int, list[int]],
        window_size: int = 50,
    ) -> ResidueStructureResult:
        """
        残基レベルのLambda³構造を計算（単一フレーム対応版）

        重要：end_frameは包含的（inclusive）として扱う！
        frames 5-5 → 1フレーム
        frames 207-208 → 2フレーム
        """
        with self.timer("compute_residue_structures"):
            # 包含的範囲として処理！
            actual_end = end_frame + 1  # スライスのために+1
            n_frames = actual_end - start_frame

            logger.info("🔬 Computing residue-level Lambda³")
            logger.info(f"   Frames: {start_frame}-{end_frame} ({n_frames} frames)")
            logger.info(f"   Residues: {len(residue_atoms)}")

            n_residues = len(residue_atoms)

            # エッジケース：空のフレーム範囲
            if n_frames <= 0:
                logger.warning(f"Empty frame range: {start_frame}-{end_frame}")
                return self._create_empty_result(n_residues)

            # 1. 残基COM計算（包含的スライス）
            with self.timer("residue_coms"):
                residue_coms = self._compute_residue_coms(
                    trajectory[start_frame:actual_end], residue_atoms
                )

            # 2. 残基レベルΛF（形状統一版）
            with self.timer("residue_lambda_f"):
                residue_lambda_f, residue_lambda_f_mag = self._compute_residue_lambda_f(
                    residue_coms
                )

            # 3. 残基レベルρT
            with self.timer("residue_rho_t"):
                residue_rho_t = self._compute_residue_rho_t(residue_coms, window_size)

            # 4. 残基間カップリング
            with self.timer("residue_coupling"):
                residue_coupling = self._compute_residue_coupling(residue_coms)

            # 結果をCPUに転送
            result = ResidueStructureResult(
                residue_lambda_f=self.to_cpu(residue_lambda_f),
                residue_lambda_f_mag=self.to_cpu(residue_lambda_f_mag),
                residue_rho_t=self.to_cpu(residue_rho_t),
                residue_coupling=self.to_cpu(residue_coupling),
                residue_coms=self.to_cpu(residue_coms),
            )

            # 形状検証
            if not result.validate_shapes():
                logger.warning("Shape validation failed! But continuing...")

            self._print_statistics_safe(result)

            return result

    def _create_empty_result(self, n_residues: int) -> ResidueStructureResult:
        """空の結果を生成（エラー回避用）"""
        return ResidueStructureResult(
            residue_lambda_f=np.zeros((0, n_residues, 3), dtype=np.float32),
            residue_lambda_f_mag=np.zeros((0, n_residues), dtype=np.float32),
            residue_rho_t=np.zeros((0, n_residues), dtype=np.float32),
            residue_coupling=np.zeros((0, n_residues, n_residues), dtype=np.float32),
            residue_coms=np.zeros((0, n_residues, 3), dtype=np.float32),
        )

    def _compute_residue_coms(
        self, trajectory: np.ndarray, residue_atoms: dict[int, list[int]]
    ) -> cp.ndarray:
        """残基COM計算（カスタムカーネル使用）"""
        n_frames, n_atoms, _ = trajectory.shape
        n_residues = len(residue_atoms)

        if self.is_gpu and HAS_GPU and residue_com_kernel is not None:
            # GPU版：カスタムカーネル使用
            trajectory_gpu = self.to_gpu(trajectory, dtype=cp.float32)
            residue_coms = residue_com_kernel(trajectory_gpu, residue_atoms)
        else:
            # CPU版フォールバック
            residue_coms = np.zeros((n_frames, n_residues, 3), dtype=np.float32)
            for res_id, atoms in residue_atoms.items():
                if len(atoms) > 0 and max(atoms) < n_atoms:
                    residue_coms[:, res_id] = np.mean(trajectory[:, atoms], axis=1)
            residue_coms = self.to_gpu(residue_coms)

        return residue_coms

    def _compute_residue_lambda_f(
        self, residue_coms: cp.ndarray
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """残基レベルΛF計算（形状統一版）"""
        n_frames, n_residues, _ = residue_coms.shape

        # 単一フレームの特別処理
        if n_frames <= 1:
            # 単一フレーム：変化なし（ゼロ）
            residue_lambda_f = self.xp.zeros(
                (n_frames, n_residues, 3), dtype=self.xp.float32
            )
            residue_lambda_f_mag = self.xp.zeros(
                (n_frames, n_residues), dtype=self.xp.float32
            )
            logger.debug("Single frame detected, returning zero lambda_f")
            return residue_lambda_f, residue_lambda_f_mag

        # 通常処理：フレーム間差分
        residue_lambda_f = self.xp.diff(residue_coms, axis=0)
        residue_lambda_f_mag = self.xp.linalg.norm(residue_lambda_f, axis=2)

        # パディング：最初のフレームを補完（diffで失われた分）
        zero_frame = self.xp.zeros((1, n_residues, 3), dtype=residue_lambda_f.dtype)
        residue_lambda_f = self.xp.concatenate([zero_frame, residue_lambda_f], axis=0)

        zero_mag = self.xp.zeros((1, n_residues), dtype=residue_lambda_f_mag.dtype)
        residue_lambda_f_mag = self.xp.concatenate(
            [zero_mag, residue_lambda_f_mag], axis=0
        )

        return residue_lambda_f, residue_lambda_f_mag

    def _compute_residue_rho_t(
        self, residue_coms: cp.ndarray, window_size: int
    ) -> cp.ndarray:
        """残基レベルρT計算（単一フレーム対応）"""
        n_frames, n_residues, _ = residue_coms.shape

        # 単一フレームの特別処理
        if n_frames == 1:
            # 単一フレーム：ゼロ分散
            logger.debug("Single frame: returning zero rho_t")
            return self.zeros((n_frames, n_residues), dtype=self.xp.float32)

        # カスタムカーネル使用
        if self.is_gpu and self.tension_kernel is not None:
            rho_t = self.zeros((n_frames, n_residues), dtype=cp.float32)

            # 2Dグリッド設定
            block_size = (16, 16)
            grid_size = (
                (n_residues + block_size[0] - 1) // block_size[0],
                (n_frames + block_size[1] - 1) // block_size[1],
            )

            self.tension_kernel(
                grid_size,
                block_size,
                (residue_coms.ravel(), rho_t, n_frames, n_residues, window_size),
            )

            return rho_t
        else:
            # CPU/GPU汎用版
            residue_rho_t = self.zeros((n_frames, n_residues))

            for frame in range(n_frames):
                for res_id in range(n_residues):
                    local_start = max(0, frame - window_size // 2)
                    local_end = min(n_frames, frame + window_size // 2 + 1)

                    local_coms = residue_coms[local_start:local_end, res_id]
                    if len(local_coms) > 1:
                        cov = self.xp.cov(local_coms.T)
                        if not self.xp.any(self.xp.isnan(cov)) and not self.xp.all(
                            cov == 0
                        ):
                            residue_rho_t[frame, res_id] = self.xp.trace(cov)

            return residue_rho_t

    def _compute_residue_coupling(self, residue_coms: cp.ndarray) -> cp.ndarray:
        """残基間カップリング計算"""
        n_frames, n_residues, _ = residue_coms.shape

        if self.is_gpu and self.coupling_kernel is not None:
            # カスタムカーネル使用（各フレーム独立）
            coupling = self.zeros((n_frames, n_residues, n_residues), dtype=cp.float32)

            # タイルサイズ
            TILE_DIM = 16
            block_size = (TILE_DIM, TILE_DIM)
            grid_size = (
                (n_residues + TILE_DIM - 1) // TILE_DIM,
                (n_residues + TILE_DIM - 1) // TILE_DIM,
            )

            # 共有メモリサイズ
            shared_mem_size = 2 * TILE_DIM * TILE_DIM * 4  # float32

            for frame in range(n_frames):
                self.coupling_kernel(
                    grid_size,
                    block_size,
                    (residue_coms[frame].ravel(), coupling[frame], n_residues),
                    shared_mem=shared_mem_size,
                )
        else:
            # CPU/GPU汎用版
            coupling = self.zeros((n_frames, n_residues, n_residues))

            for frame in range(n_frames):
                # 距離行列
                distances = self.xp.linalg.norm(
                    residue_coms[frame, :, None, :] - residue_coms[frame, None, :, :],
                    axis=2,
                )
                # カップリング
                coupling[frame] = 1.0 / (1.0 + distances)

        return coupling

    def _print_statistics_safe(self, result: ResidueStructureResult):
        """統計情報を安全に表示"""
        logger.info("  ✅ Structure computation complete")
        logger.info(f"  Frames: {result.n_frames}, Residues: {result.n_residues}")

        # 統計サマリー
        stats = result.get_summary_stats()
        if result.n_frames > 0:
            logger.info(f"  Mean ΛF: {stats['mean_lambda_f']:.3f}")
            logger.info(f"  Mean ρT: {stats['mean_rho_t']:.3f}")
            logger.info(f"  Mean Coupling: {stats['mean_coupling']:.3f}")
        else:
            logger.info("  No frames to analyze")


# ===============================
# Convenience Functions
# ===============================


def compute_residue_structures_gpu(
    trajectory: np.ndarray,
    start_frame: int,
    end_frame: int,
    residue_atoms: dict[int, list[int]],
    window_size: int = 50,
) -> ResidueStructureResult:
    """残基構造計算のラッパー関数（包含的範囲）"""
    calculator = ResidueStructuresGPU()
    return calculator.compute_residue_structures(
        trajectory, start_frame, end_frame, residue_atoms, window_size
    )


def compute_residue_lambda_f_gpu(
    residue_coms: Union[np.ndarray, cp.ndarray], backend: Optional[GPUBackend] = None
) -> tuple[np.ndarray, np.ndarray]:
    """残基ΛF計算のスタンドアロン関数（パディング済み）"""
    calculator = (
        ResidueStructuresGPU()
        if backend is None
        else ResidueStructuresGPU(device=backend.device)
    )
    coms_gpu = calculator.to_gpu(residue_coms)
    lambda_f, lambda_f_mag = calculator._compute_residue_lambda_f(coms_gpu)
    return calculator.to_cpu(lambda_f), calculator.to_cpu(lambda_f_mag)


def compute_residue_rho_t_gpu(
    residue_coms: Union[np.ndarray, cp.ndarray],
    window_size: int = 50,
    backend: Optional[GPUBackend] = None,
) -> np.ndarray:
    """残基ρT計算のスタンドアロン関数"""
    calculator = (
        ResidueStructuresGPU()
        if backend is None
        else ResidueStructuresGPU(device=backend.device)
    )
    coms_gpu = calculator.to_gpu(residue_coms)
    rho_t = calculator._compute_residue_rho_t(coms_gpu, window_size)
    return calculator.to_cpu(rho_t)


def compute_residue_coupling_gpu(
    residue_coms: Union[np.ndarray, cp.ndarray], backend: Optional[GPUBackend] = None
) -> np.ndarray:
    """残基カップリング計算のスタンドアロン関数"""
    calculator = (
        ResidueStructuresGPU()
        if backend is None
        else ResidueStructuresGPU(device=backend.device)
    )
    coms_gpu = calculator.to_gpu(residue_coms)
    coupling = calculator._compute_residue_coupling(coms_gpu)
    return calculator.to_cpu(coupling)


# ===============================
# Batch Processing
# ===============================


class ResidueStructureBatchProcessor:
    """大規模トラジェクトリのバッチ処理"""

    def __init__(
        self, calculator: ResidueStructuresGPU, batch_size: Optional[int] = None
    ):
        self.calculator = calculator
        self.batch_size = batch_size

    def process_trajectory(
        self,
        trajectory: np.ndarray,
        residue_atoms: dict[int, list[int]],
        window_size: int = 50,
        overlap: int = 100,
    ) -> ResidueStructureResult:
        """バッチ処理でトラジェクトリ全体を処理"""
        n_frames = trajectory.shape[0]

        if self.batch_size is None:
            # 自動決定
            self.batch_size = self.calculator.memory_manager.estimate_batch_size(
                trajectory.shape, dtype=np.float32
            )

        logger.info(f"Processing {n_frames} frames in batches of {self.batch_size}")

        # TODO: 実装が必要
        raise NotImplementedError("Batch processing implementation needed")
