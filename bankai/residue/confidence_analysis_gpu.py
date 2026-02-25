"""
Statistical Confidence Analysis (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

統計的信頼性をGPUで高速に評価！
ブートストラップ、順列検定、信頼区間とか全部速いよ〜！💕

by 環ちゃん
"""

import numpy as np
import logging
from typing import Optional, Union, Callable, Any
from dataclasses import dataclass

# GPU imports
try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# Local imports
from ..core import GPUBackend, GPUMemoryManager, handle_gpu_errors, profile_gpu

logger = logging.getLogger("bankai.residue.confidence")

# ===============================
# Data Classes
# ===============================


@dataclass
class ConfidenceResult:
    """信頼性解析の結果"""

    statistic_name: str  # 統計量の名前
    point_estimate: float  # 点推定値
    ci_lower: float  # 信頼区間下限
    ci_upper: float  # 信頼区間上限
    confidence_level: float  # 信頼水準
    n_bootstrap: int  # ブートストラップ回数
    p_value: Optional[float] = None  # p値
    standard_error: Optional[float] = None  # 標準誤差
    bias: Optional[float] = None  # バイアス

    @property
    def ci_width(self) -> float:
        """信頼区間の幅"""
        return self.ci_upper - self.ci_lower

    @property
    def is_significant(self) -> bool:
        """統計的有意性（0を含まない）"""
        return self.ci_lower > 0 or self.ci_upper < 0


# ===============================
# CUDA Kernels
# ===============================

# ブートストラップリサンプリングカーネル
BOOTSTRAP_RESAMPLE_KERNEL = r"""
extern "C" __global__
void bootstrap_resample_kernel(
    const float* __restrict__ data,        // (n_samples,)
    float* __restrict__ resampled,        // (n_bootstrap, n_samples)
    const int* __restrict__ indices,       // (n_bootstrap, n_samples)
    const int n_samples,
    const int n_bootstrap
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = n_bootstrap * n_samples;
    
    if (idx >= total_elements) return;
    
    const int bootstrap_idx = idx / n_samples;
    const int sample_idx = idx % n_samples;
    
    // インデックスを使ってリサンプリング
    int source_idx = indices[idx];
    resampled[idx] = data[source_idx];
}
"""

# ブートストラップ統計量計算カーネル
BOOTSTRAP_STATISTIC_KERNEL = r"""
extern "C" __global__
void compute_bootstrap_statistics_kernel(
    const float* __restrict__ resampled_x,  // (n_bootstrap, n_samples)
    const float* __restrict__ resampled_y,  // (n_bootstrap, n_samples)  
    float* __restrict__ statistics,         // (n_bootstrap,)
    const int n_samples,
    const int n_bootstrap,
    const int statistic_type  // 0: correlation, 1: mean_diff, 2: regression_slope
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= n_bootstrap) return;
    
    const float* x = &resampled_x[b * n_samples];
    const float* y = &resampled_y[b * n_samples];
    
    if (statistic_type == 0) {
        // 相関係数
        float mean_x = 0.0f, mean_y = 0.0f;
        
        // 平均
        for (int i = 0; i < n_samples; i++) {
            mean_x += x[i];
            mean_y += y[i];
        }
        mean_x /= n_samples;
        mean_y /= n_samples;
        
        // 共分散と分散
        float cov = 0.0f, var_x = 0.0f, var_y = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float dx = x[i] - mean_x;
            float dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        
        float denominator = sqrtf(var_x * var_y);
        statistics[b] = (denominator > 1e-10f) ? cov / denominator : 0.0f;
        
    } else if (statistic_type == 1) {
        // 平均差
        float mean_x = 0.0f, mean_y = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            mean_x += x[i];
            mean_y += y[i];
        }
        statistics[b] = mean_x / n_samples - mean_y / n_samples;
        
    } else if (statistic_type == 2) {
        // 回帰係数（簡易版）
        float mean_x = 0.0f, mean_y = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            mean_x += x[i];
            mean_y += y[i];
        }
        mean_x /= n_samples;
        mean_y /= n_samples;
        
        float num = 0.0f, denom = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float dx = x[i] - mean_x;
            float dy = y[i] - mean_y;
            num += dx * dy;
            denom += dx * dx;
        }
        
        statistics[b] = (denom > 1e-10f) ? num / denom : 0.0f;
    }
}
"""

# 順列検定カーネル
PERMUTATION_TEST_KERNEL = r"""
extern "C" __global__
void permutation_test_kernel(
    const float* __restrict__ group1,       // (n1,)
    const float* __restrict__ group2,       // (n2,)
    const int* __restrict__ perm_indices,   // (n_permutations, n1+n2)
    float* __restrict__ test_statistics,    // (n_permutations,)
    const int n1,
    const int n2,
    const int n_permutations
) {
    const int perm = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (perm >= n_permutations) return;
    
    const int n_total = n1 + n2;
    const int* indices = &perm_indices[perm * n_total];
    
    // 結合データから順列に従ってグループ分け
    float sum1 = 0.0f, sum2 = 0.0f;
    
    for (int i = 0; i < n1; i++) {
        int idx = indices[i];
        if (idx < n1) {
            sum1 += group1[idx];
        } else {
            sum1 += group2[idx - n1];
        }
    }
    
    for (int i = n1; i < n_total; i++) {
        int idx = indices[i];
        if (idx < n1) {
            sum2 += group1[idx];
        } else {
            sum2 += group2[idx - n1];
        }
    }
    
    // 平均差を統計量として使用
    test_statistics[perm] = sum1 / n1 - sum2 / n2;
}
"""

# ===============================
# Confidence Analyzer GPU Class
# ===============================


class ConfidenceAnalyzerGPU(GPUBackend):
    """
    統計的信頼性解析のGPU実装

    ブートストラップ法で信頼区間を計算したり、
    順列検定でp値を出したり、全部高速にできるよ〜！
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42,
        memory_manager: Optional[GPUMemoryManager] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_bootstrap : int
            ブートストラップサンプル数
        confidence_level : float
            信頼水準（0.95 = 95%）
        random_seed : int
            乱数シード
        """
        super().__init__(**kwargs)
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.memory_manager = memory_manager or GPUMemoryManager()

        # 乱数生成器
        if self.is_gpu:
            self.rng = cp.random.RandomState(seed=random_seed)
        else:
            self.rng = np.random.RandomState(seed=random_seed)

        # カーネルコンパイル
        if self.is_gpu and HAS_GPU:
            self._compile_kernels()

    def _compile_kernels(self):
        """カスタムカーネルをコンパイル"""
        try:
            self.resample_kernel = cp.RawKernel(
                BOOTSTRAP_RESAMPLE_KERNEL, "bootstrap_resample_kernel"
            )
            self.statistic_kernel = cp.RawKernel(
                BOOTSTRAP_STATISTIC_KERNEL, "compute_bootstrap_statistics_kernel"
            )
            self.permutation_kernel = cp.RawKernel(
                PERMUTATION_TEST_KERNEL, "permutation_test_kernel"
            )
            logger.debug("Confidence analysis kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile custom kernels: {e}")
            self.resample_kernel = None
            self.statistic_kernel = None
            self.permutation_kernel = None

    @handle_gpu_errors
    @profile_gpu
    def analyze(
        self,
        causality_chains: list[tuple[int, int, float]],
        anomaly_scores: dict[int, np.ndarray],
        n_bootstrap: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        因果連鎖の信頼性を解析（メインエントリーポイント）

        two_stage_analyzer_gpu.pyから呼ばれる汎用解析メソッド。
        因果連鎖の各ペアについて相関の信頼区間を計算する。

        Parameters
        ----------
        causality_chains : List[Tuple[int, int, float]]
            因果連鎖のリスト [(from_res, to_res, strength), ...]
        anomaly_scores : Dict[int, np.ndarray]
            残基ID -> 異常スコアの時系列
        n_bootstrap : int, optional
            ブートストラップ回数（高速化のため少なめのデフォルト）

        Returns
        -------
        List[Dict[str, Any]]
            各因果ペアの信頼性解析結果
        """
        if n_bootstrap is None:
            n_bootstrap = min(self.n_bootstrap, 50)  # 高速化のため少なめ

        results = []

        # 各因果ペアを解析
        for pair_idx, (from_res, to_res, strength) in enumerate(causality_chains):
            # 両方の残基のスコアが存在する場合のみ解析
            if from_res not in anomaly_scores or to_res not in anomaly_scores:
                logger.debug(
                    f"Skipping pair ({from_res}, {to_res}): missing anomaly scores"
                )
                continue

            try:
                # 時系列の長さを確認
                series_x = anomaly_scores[from_res]
                series_y = anomaly_scores[to_res]

                # 長さを合わせる（短い方に合わせる）
                min_len = min(len(series_x), len(series_y))
                if min_len < 10:  # 最低10点は必要
                    logger.warning(
                        f"Series too short for pair ({from_res}, {to_res}): {min_len} points"
                    )
                    continue

                series_x = series_x[:min_len]
                series_y = series_y[:min_len]

                # 相関の信頼区間を計算
                conf_result = self.bootstrap_correlation_confidence(
                    series_x, series_y, n_bootstrap=n_bootstrap
                )

                # 結果を構築
                result_dict = {
                    "pair_index": pair_idx,
                    "from_res": from_res,
                    "to_res": to_res,
                    "original_strength": float(strength),
                    "correlation": float(conf_result.point_estimate),
                    "ci_lower": float(conf_result.ci_lower),
                    "ci_upper": float(conf_result.ci_upper),
                    "is_significant": bool(conf_result.is_significant),
                    "standard_error": float(conf_result.standard_error)
                    if conf_result.standard_error
                    else None,
                    "bias": float(conf_result.bias) if conf_result.bias else None,
                }

                results.append(result_dict)

            except Exception as e:
                logger.warning(
                    f"Confidence analysis failed for pair ({from_res}, {to_res}): {e}"
                )
                continue

        # サマリー情報を追加
        if results:
            n_significant = sum(1 for r in results if r["is_significant"])
            logger.info(
                f"Confidence analysis complete: {len(results)} pairs analyzed, "
                f"{n_significant} significant"
            )

        return results

    @handle_gpu_errors
    @profile_gpu
    def bootstrap_correlation_confidence(
        self,
        series_x: np.ndarray,
        series_y: np.ndarray,
        n_bootstrap: Optional[int] = None,
    ) -> ConfidenceResult:
        """
        相関係数のブートストラップ信頼区間

        Parameters
        ----------
        series_x : np.ndarray
            系列X
        series_y : np.ndarray
            系列Y
        n_bootstrap : int, optional
            ブートストラップ回数

        Returns
        -------
        ConfidenceResult
            信頼区間の結果
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap

        n_samples = len(series_x)

        # GPU転送
        x_gpu = self.to_gpu(series_x)
        y_gpu = self.to_gpu(series_y)

        # 元の相関係数（エラーハンドリング付き）
        try:
            corr_matrix = self.xp.corrcoef(x_gpu, y_gpu)
            original_corr = float(corr_matrix[0, 1])

            # NaNチェック
            if self.xp.isnan(original_corr):
                logger.warning("Original correlation is NaN, returning default result")
                return ConfidenceResult(
                    statistic_name="correlation",
                    point_estimate=0.0,
                    ci_lower=0.0,
                    ci_upper=0.0,
                    confidence_level=self.confidence_level,
                    n_bootstrap=0,
                    standard_error=0.0,
                    bias=0.0,
                )
        except Exception as e:
            logger.warning(f"Failed to compute correlation: {e}")
            return ConfidenceResult(
                statistic_name="correlation",
                point_estimate=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                confidence_level=self.confidence_level,
                n_bootstrap=0,
                standard_error=0.0,
                bias=0.0,
            )

        # ブートストラップ
        with self.timer("bootstrap"):
            bootstrap_correlations = self._bootstrap_statistic(
                x_gpu, y_gpu, lambda x, y: self.xp.corrcoef(x, y)[0, 1], n_bootstrap
            )

        # 信頼区間計算
        ci_lower, ci_upper = self._compute_confidence_interval(
            bootstrap_correlations, self.confidence_level
        )

        # バイアス補正
        bias = float(self.xp.mean(bootstrap_correlations)) - original_corr

        # 標準誤差
        se = float(self.xp.std(bootstrap_correlations))

        result = ConfidenceResult(
            statistic_name="correlation",
            point_estimate=original_corr,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            confidence_level=self.confidence_level,
            n_bootstrap=n_bootstrap,
            standard_error=se,
            bias=bias,
        )

        return result

    def _bootstrap_statistic(
        self,
        x_gpu: Union[np.ndarray, cp.ndarray],
        y_gpu: Union[np.ndarray, cp.ndarray],
        statistic_func: Callable,
        n_bootstrap: int,
    ) -> Union[np.ndarray, cp.ndarray]:
        """汎用ブートストラップ（エラーハンドリング強化）"""
        n_samples = len(x_gpu)
        bootstrap_stats = self.zeros(n_bootstrap)

        # バッチ処理でメモリ効率化
        batch_size = min(100, n_bootstrap)

        for i in range(0, n_bootstrap, batch_size):
            batch_end = min(i + batch_size, n_bootstrap)
            batch_n = batch_end - i

            # リサンプリングインデックス生成
            indices = self.rng.randint(0, n_samples, size=(batch_n, n_samples))

            # バッチで統計量計算
            for j in range(batch_n):
                try:
                    x_resampled = x_gpu[indices[j]]
                    y_resampled = y_gpu[indices[j]]
                    stat_value = statistic_func(x_resampled, y_resampled)

                    # NaNチェック
                    if not self.xp.isnan(stat_value):
                        bootstrap_stats[i + j] = stat_value
                    else:
                        bootstrap_stats[i + j] = 0.0

                except Exception as e:
                    logger.debug(f"Bootstrap iteration {i + j} failed: {e}")
                    bootstrap_stats[i + j] = 0.0

        return bootstrap_stats

    def _compute_confidence_interval(
        self, bootstrap_stats: Union[np.ndarray, cp.ndarray], confidence_level: float
    ) -> tuple[float, float]:
        """信頼区間計算（パーセンタイル法）"""
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        # NaN除去
        valid_stats = bootstrap_stats[~self.xp.isnan(bootstrap_stats)]

        if len(valid_stats) == 0:
            return 0.0, 0.0

        ci_lower = self.xp.percentile(valid_stats, lower_percentile)
        ci_upper = self.xp.percentile(valid_stats, upper_percentile)

        return ci_lower, ci_upper

    @handle_gpu_errors
    @profile_gpu
    def permutation_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        n_permutations: int = 10000,
        alternative: str = "two-sided",
    ) -> float:
        """
        順列検定でp値を計算

        Parameters
        ----------
        group1 : np.ndarray
            グループ1のデータ
        group2 : np.ndarray
            グループ2のデータ
        n_permutations : int
            順列数
        alternative : str
            対立仮説（'two-sided', 'greater', 'less'）

        Returns
        -------
        float
            p値
        """
        # GPU転送
        group1_gpu = self.to_gpu(group1)
        group2_gpu = self.to_gpu(group2)

        n1 = len(group1_gpu)
        n2 = len(group2_gpu)

        # 観測統計量（平均差）
        observed_stat = float(self.xp.mean(group1_gpu) - self.xp.mean(group2_gpu))

        # 結合データ
        combined = self.xp.concatenate([group1_gpu, group2_gpu])
        n_total = n1 + n2

        # 順列統計量
        perm_stats = self.zeros(n_permutations)

        # バッチ処理
        batch_size = min(1000, n_permutations)

        for i in range(0, n_permutations, batch_size):
            batch_end = min(i + batch_size, n_permutations)
            batch_n = batch_end - i

            for j in range(batch_n):
                # ランダム順列
                perm = self.rng.permutation(n_total)
                perm_group1 = combined[perm[:n1]]
                perm_group2 = combined[perm[n1:]]

                # 統計量計算
                perm_stats[i + j] = self.xp.mean(perm_group1) - self.xp.mean(
                    perm_group2
                )

        # p値計算
        if alternative == "two-sided":
            p_value = self.xp.mean(self.xp.abs(perm_stats) >= abs(observed_stat))
        elif alternative == "greater":
            p_value = self.xp.mean(perm_stats >= observed_stat)
        else:  # less
            p_value = self.xp.mean(perm_stats <= observed_stat)

        return float(p_value)

    @profile_gpu
    def compute_confidence_intervals(
        self,
        data: np.ndarray,
        statistics: list[str] = ["mean", "std", "median"],
        n_bootstrap: Optional[int] = None,
    ) -> dict[str, ConfidenceResult]:
        """
        複数統計量の信頼区間を一度に計算

        Parameters
        ----------
        data : np.ndarray
            データ
        statistics : list of str
            計算する統計量
        n_bootstrap : int, optional
            ブートストラップ回数

        Returns
        -------
        dict
            統計量名 -> ConfidenceResult
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap

        # GPU転送
        data_gpu = self.to_gpu(data)
        n_samples = len(data_gpu)

        results = {}

        # 統計量関数マップ
        stat_funcs = {
            "mean": self.xp.mean,
            "std": self.xp.std,
            "median": self.xp.median,
            "var": self.xp.var,
            "min": self.xp.min,
            "max": self.xp.max,
        }

        for stat_name in statistics:
            if stat_name not in stat_funcs:
                logger.warning(f"Unknown statistic: {stat_name}")
                continue

            stat_func = stat_funcs[stat_name]

            # 元の統計量
            original_stat = float(stat_func(data_gpu))

            # ブートストラップ
            bootstrap_stats = self.zeros(n_bootstrap)

            for i in range(n_bootstrap):
                # リサンプリング
                indices = self.rng.randint(0, n_samples, size=n_samples)
                resampled = data_gpu[indices]
                bootstrap_stats[i] = stat_func(resampled)

            # 信頼区間
            ci_lower, ci_upper = self._compute_confidence_interval(
                bootstrap_stats, self.confidence_level
            )

            # 結果作成
            results[stat_name] = ConfidenceResult(
                statistic_name=stat_name,
                point_estimate=original_stat,
                ci_lower=float(ci_lower),
                ci_upper=float(ci_upper),
                confidence_level=self.confidence_level,
                n_bootstrap=n_bootstrap,
                standard_error=float(self.xp.std(bootstrap_stats)),
                bias=float(self.xp.mean(bootstrap_stats)) - original_stat,
            )

        return results

    def evaluate_statistical_significance(
        self,
        test_statistic: float,
        null_distribution: np.ndarray,
        alternative: str = "two-sided",
    ) -> tuple[float, bool]:
        """
        統計的有意性を評価

        Parameters
        ----------
        test_statistic : float
            検定統計量
        null_distribution : np.ndarray
            帰無分布
        alternative : str
            対立仮説

        Returns
        -------
        p_value : float
            p値
        is_significant : bool
            有意かどうか（α=0.05）
        """
        # GPU転送
        null_dist_gpu = self.to_gpu(null_distribution)

        # p値計算
        if alternative == "two-sided":
            p_value = self.xp.mean(self.xp.abs(null_dist_gpu) >= abs(test_statistic))
        elif alternative == "greater":
            p_value = self.xp.mean(null_dist_gpu >= test_statistic)
        else:  # less
            p_value = self.xp.mean(null_dist_gpu <= test_statistic)

        p_value = float(p_value)
        is_significant = p_value < 0.05

        return p_value, is_significant


# ===============================
# Standalone Functions
# ===============================


def bootstrap_correlation_confidence_gpu(
    series_x: np.ndarray,
    series_y: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    **kwargs,
) -> ConfidenceResult:
    """相関係数の信頼区間計算のスタンドアロン関数"""
    analyzer = ConfidenceAnalyzerGPU(
        n_bootstrap=n_bootstrap, confidence_level=confidence_level, **kwargs
    )
    return analyzer.bootstrap_correlation_confidence(series_x, series_y)


def permutation_test_gpu(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    **kwargs,
) -> float:
    """順列検定のスタンドアロン関数"""
    analyzer = ConfidenceAnalyzerGPU(**kwargs)
    return analyzer.permutation_test(group1, group2, n_permutations, alternative)


def compute_confidence_intervals_gpu(
    data: np.ndarray,
    statistics: list[str] = ["mean", "std"],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    **kwargs,
) -> dict[str, ConfidenceResult]:
    """複数統計量の信頼区間計算のスタンドアロン関数"""
    analyzer = ConfidenceAnalyzerGPU(
        n_bootstrap=n_bootstrap, confidence_level=confidence_level, **kwargs
    )
    return analyzer.compute_confidence_intervals(data, statistics)


def evaluate_statistical_significance_gpu(
    test_statistic: float,
    null_distribution: np.ndarray,
    alternative: str = "two-sided",
    backend: Optional[GPUBackend] = None,
) -> tuple[float, bool]:
    """統計的有意性評価のスタンドアロン関数"""
    analyzer = ConfidenceAnalyzerGPU()
    if backend:
        analyzer.device = backend.device
        analyzer.is_gpu = backend.is_gpu
        analyzer.xp = backend.xp

    return analyzer.evaluate_statistical_significance(
        test_statistic, null_distribution, alternative
    )


# ===============================
# Utility Functions
# ===============================


def create_null_distribution_gpu(
    data: np.ndarray,
    statistic_func: Callable,
    n_samples: int = 10000,
    backend: Optional[GPUBackend] = None,
) -> np.ndarray:
    """
    帰無分布を生成（ランダム化で）

    Parameters
    ----------
    data : np.ndarray
        元データ
    statistic_func : callable
        統計量関数
    n_samples : int
        サンプル数

    Returns
    -------
    np.ndarray
        帰無分布
    """
    backend = backend or GPUBackend()
    data_gpu = backend.to_gpu(data)
    null_dist = backend.zeros(n_samples)

    rng = (
        cp.random.RandomState(seed=42)
        if backend.is_gpu
        else np.random.RandomState(seed=42)
    )

    for i in range(n_samples):
        # データをシャッフル
        shuffled = data_gpu.copy()
        rng.shuffle(shuffled)

        # 統計量計算
        null_dist[i] = statistic_func(shuffled)

    return backend.to_cpu(null_dist)
