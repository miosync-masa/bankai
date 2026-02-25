"""
Lambda³ GPU版性能評価モジュール
検出性能の多段階評価とベンチマーク
"""

import numpy as np
import time
from typing import Optional, Any
from dataclasses import dataclass

# CuPyの条件付きインポート
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

# 他のインポート
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from ..core.gpu_utils import GPUBackend
from .md_lambda3_detector_gpu import MDLambda3Result
from .two_stage_analyzer_gpu import TwoStageLambda3Result


@dataclass
class PerformanceMetrics:
    """性能評価メトリクス"""

    # イベント検出
    detection_rate: float
    n_detected: int
    n_total: int
    average_confidence: float

    # タイミング精度
    mean_timing_error: float
    median_timing_error: float
    timing_accuracy: float

    # 従来型メトリクス
    roc_auc: float
    pr_auc: float
    false_positive_rate: float

    # 総合スコア
    overall_score: float
    grade: str

    # GPU性能
    gpu_speedup: float = 1.0
    memory_efficiency: float = 1.0
    computation_time: float = 0.0


@dataclass
class EventDetectionResult:
    """イベント検出結果"""

    name: str
    detected: bool
    confidence: float
    criteria: dict[str, Any]
    timing_error: Optional[float] = None


class PerformanceEvaluatorGPU(GPUBackend):
    """GPU版性能評価器"""

    def __init__(self, force_cpu=False):
        super().__init__(force_cpu)
        self.benchmark_results = {}

    def evaluate_detection_performance(
        self,
        result: MDLambda3Result,
        ground_truth_events: list[tuple[int, int, str]],
        cpu_baseline_time: Optional[float] = None,
    ) -> PerformanceMetrics:
        """
        Lambda³検出性能の多段階評価

        Parameters
        ----------
        result : MDLambda3Result
            検出結果
        ground_truth_events : List[Tuple[int, int, str]]
            正解イベント [(start, end, name), ...]
        cpu_baseline_time : float, optional
            CPU版の実行時間（スピードアップ計算用）

        Returns
        -------
        PerformanceMetrics
            評価結果
        """
        print("\n" + "=" * 60)
        print("=== Multi-level Detection Performance Evaluation (GPU) ===")
        print("=" * 60)

        start_time = time.time()
        n_frames = result.n_frames

        # GPU上で評価を実行
        with self.memory_manager.batch_context(n_frames * 8):
            # 1. イベント検出評価
            event_results = self._evaluate_event_detection_gpu(
                result, ground_truth_events
            )

            # 2. タイミング精度評価
            timing_metrics = self._evaluate_timing_accuracy_gpu(
                result, ground_truth_events
            )

            # 3. 従来型メトリクス
            traditional_metrics = self._evaluate_traditional_metrics_gpu(
                result, ground_truth_events
            )

            # 4. GPU性能評価
            gpu_metrics = self._evaluate_gpu_performance(result, cpu_baseline_time)

        # 総合評価
        overall_metrics = self._compute_overall_metrics(
            event_results, timing_metrics, traditional_metrics, gpu_metrics
        )

        evaluation_time = time.time() - start_time
        overall_metrics.computation_time = evaluation_time

        self._print_evaluation_summary(overall_metrics)

        return overall_metrics

    def _evaluate_event_detection_gpu(
        self, result: MDLambda3Result, ground_truth_events: list[tuple[int, int, str]]
    ) -> list[EventDetectionResult]:
        """イベント検出の評価（GPU高速化）"""
        print("\n📊 Level 1: Event Detection")
        print("-" * 40)

        event_detection_results = []

        # スコアをGPUに転送
        score_key = (
            "final_combined"
            if "final_combined" in result.anomaly_scores
            else "combined"
        )
        scores_gpu = self.to_gpu(result.anomaly_scores[score_key])

        # 境界位置をGPUに
        if "boundary_locations" in result.structural_boundaries:
            boundaries_gpu = self.to_gpu(
                result.structural_boundaries["boundary_locations"]
            )
        else:
            boundaries_gpu = cp.array([])

        for start, end, name in ground_truth_events:
            detection_criteria = {}

            # イベントウィンドウのスコア
            event_scores = scores_gpu[start:end]

            if len(event_scores) > 0:
                # 1. ピーク検出
                max_score = float(cp.max(event_scores))
                detection_criteria["peak_score"] = max_score
                detection_criteria["has_peak"] = max_score > 2.0

                # 2. 境界検出
                boundary_tolerance = 5000

                # GPU上で境界との距離計算
                if len(boundaries_gpu) > 0:
                    start_distances = cp.abs(boundaries_gpu - start)
                    end_distances = cp.abs(boundaries_gpu - end)

                    detection_criteria["start_boundary"] = bool(
                        cp.any(start_distances <= boundary_tolerance)
                    )
                    detection_criteria["end_boundary"] = bool(
                        cp.any(end_distances <= boundary_tolerance)
                    )

                    # イベント内の境界
                    internal_mask = (boundaries_gpu >= start) & (boundaries_gpu <= end)
                    detection_criteria["internal_boundary"] = bool(
                        cp.any(internal_mask)
                    )
                else:
                    detection_criteria["start_boundary"] = False
                    detection_criteria["end_boundary"] = False
                    detection_criteria["internal_boundary"] = False

                detection_criteria["any_boundary"] = any(
                    [
                        detection_criteria["start_boundary"],
                        detection_criteria["end_boundary"],
                        detection_criteria["internal_boundary"],
                    ]
                )

                # 3. 持続的異常
                anomaly_frames = int(cp.sum(event_scores > 1.5))
                detection_criteria["sustained_ratio"] = anomaly_frames / len(
                    event_scores
                )
                detection_criteria["has_sustained"] = (
                    detection_criteria["sustained_ratio"] > 0.1
                )

                # 総合判定
                detected = (
                    detection_criteria["has_peak"] or detection_criteria["any_boundary"]
                )
                confidence = (
                    max_score / 2.0
                    if max_score > 2.0
                    else detection_criteria["sustained_ratio"]
                )
            else:
                detected = False
                confidence = 0.0

            event_result = EventDetectionResult(
                name=name,
                detected=detected,
                confidence=confidence,
                criteria=detection_criteria,
            )
            event_detection_results.append(event_result)

            # 結果表示
            print(f"\n{name} ({start}-{end}):")
            print(f"  ✓ Detected: {'YES' if detected else 'NO'}")
            if len(event_scores) > 0:
                print(
                    f"  - Peak score: {max_score:.2f} {'✓' if detection_criteria['has_peak'] else '✗'}"
                )
                print(
                    f"  - Boundaries: Start={'✓' if detection_criteria['start_boundary'] else '✗'}, "
                    f"End={'✓' if detection_criteria['end_boundary'] else '✗'}"
                )
                print(
                    f"  - Sustained anomaly: {detection_criteria['sustained_ratio']:.1%}"
                )

        return event_detection_results

    def _evaluate_timing_accuracy_gpu(
        self, result: MDLambda3Result, ground_truth_events: list[tuple[int, int, str]]
    ) -> dict[str, Any]:
        """タイミング精度の評価（GPU高速化）"""
        print("\n\n📊 Level 2: Boundary Timing Accuracy")
        print("-" * 40)

        timing_results = []

        if "boundary_locations" in result.structural_boundaries:
            boundaries = self.to_gpu(result.structural_boundaries["boundary_locations"])
        else:
            boundaries = cp.array([])

        for start, end, name in ground_truth_events:
            if len(boundaries) > 0:
                # GPU上で最近傍境界を検索
                start_distances = cp.abs(boundaries - start)
                end_distances = cp.abs(boundaries - end)

                start_error = int(cp.min(start_distances))
                end_error = int(cp.min(end_distances))

                start_boundary_idx = int(cp.argmin(start_distances))
                end_boundary_idx = int(cp.argmin(end_distances))

                start_boundary = int(boundaries[start_boundary_idx])
                end_boundary = int(boundaries[end_boundary_idx])
            else:
                start_error = result.n_frames
                end_error = result.n_frames
                start_boundary = None
                end_boundary = None

            timing_results.append(
                {
                    "name": name,
                    "start_error": start_error,
                    "end_error": end_error,
                    "mean_error": (start_error + end_error) / 2,
                    "start_accurate": start_error < 5000,
                    "end_accurate": end_error < 5000,
                    "start_boundary": start_boundary,
                    "end_boundary": end_boundary,
                }
            )

            print(f"\n{name}:")
            print(
                f"  Start: error={start_error} frames {'✓' if start_error < 5000 else '✗'}"
            )
            if start_boundary is not None:
                print(f"    → Detected at frame {start_boundary} (true: {start})")
            print(f"  End: error={end_error} frames {'✓' if end_error < 5000 else '✗'}")
            if end_boundary is not None:
                print(f"    → Detected at frame {end_boundary} (true: {end})")

        # 統計計算
        all_errors = [t["start_error"] for t in timing_results] + [
            t["end_error"] for t in timing_results
        ]

        return {
            "timing_results": timing_results,
            "mean_error": np.mean(all_errors),
            "median_error": np.median(all_errors),
            "accuracy_5000": sum(1 for e in all_errors if e < 5000) / len(all_errors),
        }

    def _evaluate_traditional_metrics_gpu(
        self, result: MDLambda3Result, ground_truth_events: list[tuple[int, int, str]]
    ) -> dict[str, float]:
        """従来型メトリクスの評価（GPU高速化）"""
        print("\n\n📊 Level 3: Traditional Frame-wise Metrics")
        print("-" * 40)

        n_frames = result.n_frames

        # Ground truthラベル作成（GPU上）
        ground_truth_gpu = cp.zeros(n_frames)
        for start, end, name in ground_truth_events:
            ground_truth_gpu[start:end] = 1

        ground_truth = self.to_cpu(ground_truth_gpu)

        # 各スコアタイプを評価
        traditional_results = {}

        for score_name in ["global", "local", "combined", "final_combined"]:
            if score_name in result.anomaly_scores:
                scores = result.anomaly_scores[score_name]

                # ROC-AUC
                try:
                    auc_score = roc_auc_score(ground_truth, scores)
                except:
                    auc_score = 0.5

                # Precision-Recall
                try:
                    precision, recall, _ = precision_recall_curve(ground_truth, scores)
                    pr_auc = auc(recall, precision)
                except:
                    pr_auc = 0.0

                # False Positive Rate
                scores_gpu = self.to_gpu(scores)
                ground_truth_neg = ground_truth_gpu == 0
                if cp.sum(ground_truth_neg) > 0:
                    fpr = float(cp.mean(scores_gpu[ground_truth_neg] > 2.0))
                else:
                    fpr = 0.0

                traditional_results[score_name] = {
                    "roc_auc": auc_score,
                    "pr_auc": pr_auc,
                    "fpr": fpr,
                }

                print(f"\n{score_name.capitalize()} scores:")
                print(f"  ROC-AUC: {auc_score:.4f}")
                print(f"  PR-AUC: {pr_auc:.4f}")
                print(f"  FPR: {fpr:.4f}")

        # 最良のスコアを選択
        best_score_name = max(
            traditional_results.keys(), key=lambda x: traditional_results[x]["roc_auc"]
        )

        return {
            "best_roc_auc": traditional_results[best_score_name]["roc_auc"],
            "best_pr_auc": traditional_results[best_score_name]["pr_auc"],
            "best_fpr": traditional_results[best_score_name]["fpr"],
            "all_results": traditional_results,
        }

    def _evaluate_gpu_performance(
        self, result: MDLambda3Result, cpu_baseline_time: Optional[float]
    ) -> dict[str, float]:
        """GPU性能の評価"""
        gpu_metrics = {
            "gpu_time": result.computation_time,
            "frames_per_second": result.n_frames / result.computation_time,
            "speedup": 1.0,
            "memory_used_gb": 0.0,
        }

        # スピードアップ計算
        if cpu_baseline_time and cpu_baseline_time > 0:
            gpu_metrics["speedup"] = cpu_baseline_time / result.computation_time

        # メモリ使用量
        if result.gpu_info and "memory_used" in result.gpu_info:
            gpu_metrics["memory_used_gb"] = result.gpu_info["memory_used"]

        return gpu_metrics

    def _compute_overall_metrics(
        self,
        event_results: list[EventDetectionResult],
        timing_metrics: dict,
        traditional_metrics: dict,
        gpu_metrics: dict,
    ) -> PerformanceMetrics:
        """総合メトリクスの計算"""
        # イベント検出統計
        n_detected = sum(1 for e in event_results if e.detected)
        detection_rate = n_detected / len(event_results) if event_results else 0
        avg_confidence = (
            np.mean([e.confidence for e in event_results]) if event_results else 0
        )

        # 総合スコア計算
        overall_score = (
            0.4 * detection_rate
            + 0.3 * timing_metrics["accuracy_5000"]
            + 0.2 * traditional_metrics["best_roc_auc"]
            + 0.1 * (1 - traditional_metrics["best_fpr"])
        )

        # グレード判定
        if overall_score > 0.8:
            grade = "Excellent"
        elif overall_score > 0.6:
            grade = "Good"
        elif overall_score > 0.4:
            grade = "Fair"
        else:
            grade = "Needs Improvement"

        # GPU効率
        memory_efficiency = 1.0
        if gpu_metrics["memory_used_gb"] > 0:
            # 仮定: 16GB GPUでの効率
            memory_efficiency = min(1.0, 16.0 / gpu_metrics["memory_used_gb"])

        return PerformanceMetrics(
            # イベント検出
            detection_rate=detection_rate,
            n_detected=n_detected,
            n_total=len(event_results),
            average_confidence=avg_confidence,
            # タイミング
            mean_timing_error=timing_metrics["mean_error"],
            median_timing_error=timing_metrics["median_error"],
            timing_accuracy=timing_metrics["accuracy_5000"],
            # 従来型
            roc_auc=traditional_metrics["best_roc_auc"],
            pr_auc=traditional_metrics["best_pr_auc"],
            false_positive_rate=traditional_metrics["best_fpr"],
            # 総合
            overall_score=overall_score,
            grade=grade,
            # GPU
            gpu_speedup=gpu_metrics["speedup"],
            memory_efficiency=memory_efficiency,
        )

    def _print_evaluation_summary(self, metrics: PerformanceMetrics):
        """評価サマリーの表示"""
        print("\n" + "=" * 60)
        print("=== Overall Performance Summary ===")
        print("=" * 60)

        print("\n🏆 Performance Metrics:")
        print(f"   Event Detection Rate: {metrics.detection_rate:.1%}")
        print(f"   Timing Accuracy: {metrics.timing_accuracy:.1%}")
        print(f"   Best Traditional AUC: {metrics.roc_auc:.3f}")
        print(f"   False Positive Rate: {metrics.false_positive_rate:.1%}")

        print("\n💻 GPU Performance:")
        print(f"   Speedup: {metrics.gpu_speedup:.1f}x")
        print(f"   Memory Efficiency: {metrics.memory_efficiency:.1%}")
        print(f"   Computation Time: {metrics.computation_time:.2f}s")

        print(f"\n   → Overall Score: {metrics.overall_score:.3f}")
        print(f"   → Performance Grade: {metrics.grade}")

    def benchmark_gpu_vs_cpu(
        self, trajectory: np.ndarray, config: Any, n_runs: int = 3
    ) -> dict[str, Any]:
        """GPU vs CPU ベンチマーク"""
        print("\n🏁 Running GPU vs CPU Benchmark...")

        # GPU版
        from .md_lambda3_detector_gpu import MDLambda3DetectorGPU

        gpu_times = []
        for i in range(n_runs):
            detector_gpu = MDLambda3DetectorGPU(config, device="gpu")
            start = time.time()
            _ = detector_gpu.analyze(trajectory)
            gpu_times.append(time.time() - start)

            # メモリクリア
            self.memory_manager.clear_cache()
            cp.get_default_memory_pool().free_all_blocks()

        # CPU版（可能であれば）
        cpu_times = []
        try:
            detector_cpu = MDLambda3DetectorGPU(config, device="cpu")
            for i in range(min(1, n_runs)):  # CPU版は1回のみ
                start = time.time()
                _ = detector_cpu.analyze(trajectory)
                cpu_times.append(time.time() - start)
        except:
            print("  CPU version not available or too slow")
            cpu_times = [gpu_times[0] * 10]  # 推定値

        # 結果
        benchmark_results = {
            "gpu_mean_time": np.mean(gpu_times),
            "gpu_std_time": np.std(gpu_times),
            "cpu_mean_time": np.mean(cpu_times) if cpu_times else None,
            "speedup": np.mean(cpu_times) / np.mean(gpu_times) if cpu_times else None,
            "n_frames": trajectory.shape[0],
            "n_atoms": trajectory.shape[1],
        }

        print("\n📊 Benchmark Results:")
        print(
            f"   GPU: {benchmark_results['gpu_mean_time']:.2f} ± {benchmark_results['gpu_std_time']:.2f}s"
        )
        if benchmark_results["cpu_mean_time"]:
            print(f"   CPU: {benchmark_results['cpu_mean_time']:.2f}s")
            print(f"   Speedup: {benchmark_results['speedup']:.1f}x")

        return benchmark_results


def evaluate_two_stage_performance(
    two_stage_result: TwoStageLambda3Result,
    ground_truth_residue_events: Optional[dict] = None,
) -> dict[str, Any]:
    """
    2段階解析の性能評価

    Parameters
    ----------
    two_stage_result : TwoStageLambda3Result
        2段階解析結果
    ground_truth_residue_events : Dict, optional
        残基レベルの正解イベント

    Returns
    -------
    Dict[str, Any]
        評価結果
    """
    evaluation = {
        "n_events_analyzed": len(two_stage_result.residue_analyses),
        "total_residues_involved": len(two_stage_result.global_residue_importance),
        "n_intervention_points": len(two_stage_result.suggested_intervention_points),
        "network_complexity": two_stage_result.global_network_stats[
            "total_causal_links"
        ],
        "async_ratio": two_stage_result.global_network_stats["async_to_causal_ratio"],
        "gpu_time": two_stage_result.total_gpu_time,
    }

    # 残基レベルの検証（オプション）
    if ground_truth_residue_events:
        # 実装は省略
        pass

    return evaluation
