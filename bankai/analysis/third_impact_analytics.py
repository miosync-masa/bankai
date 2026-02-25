#!/usr/bin/env python3
"""
Third Impact Analytics v3.0 - GPU Atomic Network Edition
========================================================

原子レベル量子痕跡の高速検出 + 高度なネットワーク解析
residue_networkの賢い判定を原子レベルで実現！！

- 単一フレーム：瞬間的協調ネットワーク
- 複数フレーム：非同期相関・ラグ付き因果・適応的窓
- 残基間ブリッジ：原子が担う残基間情報伝達を特定

Version: 3.0.0 - GPU Atomic Network Edition
Authors: 環ちゃん & ご主人さま 💕
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from collections import Counter, defaultdict

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy.spatial.distance import cdist as cp_cdist

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

logger = logging.getLogger("lambda3_gpu.analysis.third_impact")

# ============================================
# Data Classes (v3.0 拡張版)
# ============================================


@dataclass
class AtomicQuantumTrace:
    """原子レベル量子痕跡（ネットワーク情報付き）"""

    atom_id: int
    residue_id: int

    # 統計的異常度
    displacement_zscore: float = 0.0
    lambda_change: float = 0.0

    # 量子シグネチャー
    quantum_signature: str = "unknown"
    confidence: float = 0.0

    # ネットワーク特性（NEW!）
    connectivity_degree: int = 0  # 接続数
    is_hub: bool = False  # ハブ原子か
    is_bridge: bool = False  # 残基間ブリッジか


@dataclass
class AtomicNetworkLink:
    """原子間ネットワークリンク"""

    from_atom: int
    to_atom: int
    from_residue: int
    to_residue: int
    link_type: str  # 'sync', 'causal', 'async'
    strength: float
    lag: int = 0
    distance: Optional[float] = None


@dataclass
class ResidueBridge:
    """残基間ブリッジ情報"""

    from_residue: int
    to_residue: int
    bridge_atoms: List[Tuple[int, int]]  # (from_atom, to_atom)のリスト
    total_strength: float
    dominant_type: str  # 'sync' or 'causal'


@dataclass
class EventOrigin:
    """イベント起源情報（ネットワーク付き）"""

    genesis_atoms: List[int] = field(default_factory=list)
    first_wave_atoms: List[int] = field(default_factory=list)

    # ネットワーク起源（NEW!）
    network_initiators: List[int] = field(default_factory=list)  # ネットワークハブ

    # 統計情報
    mean_displacement: float = 0.0
    std_displacement: float = 0.0
    threshold_used: float = 0.0


@dataclass
class AtomicNetworkResult:
    """原子ネットワーク解析結果"""

    sync_network: List[AtomicNetworkLink] = field(default_factory=list)
    causal_network: List[AtomicNetworkLink] = field(default_factory=list)
    async_network: List[AtomicNetworkLink] = field(default_factory=list)

    residue_bridges: List[ResidueBridge] = field(default_factory=list)
    adaptive_windows: Dict[int, int] = field(default_factory=dict)

    network_pattern: str = "unknown"  # 'instantaneous', 'parallel', 'cascade'
    hub_atoms: List[int] = field(default_factory=list)


@dataclass
class ThirdImpactResult:
    """Third Impact解析結果（v3.0）"""

    event_name: str
    residue_id: int
    event_type: str

    # 起源情報
    origin: EventOrigin = field(default_factory=EventOrigin)

    # 原子痕跡
    quantum_atoms: Dict[int, AtomicQuantumTrace] = field(default_factory=dict)

    # 原子ネットワーク（NEW!）
    atomic_network: Optional[AtomicNetworkResult] = None

    # 統計
    n_quantum_atoms: int = 0
    n_network_links: int = 0
    n_residue_bridges: int = 0
    strongest_signature: str = ""
    max_confidence: float = 0.0

    # 創薬ターゲット
    drug_target_atoms: List[int] = field(default_factory=list)
    bridge_target_atoms: List[int] = field(default_factory=list)  # ブリッジ原子


# ============================================
# GPU Atomic Network Analyzer
# ============================================


class AtomicNetworkGPU:
    """
    原子レベルネットワーク解析（GPU高速化版）
    residue_networkの高度な手法を原子レベルに適用！
    """

    def __init__(
        self,
        correlation_threshold: float = 0.6,
        sync_threshold: float = 0.8,
        max_lag: int = 5,
        distance_cutoff: float = 5.0,
    ):
        """
        Parameters
        ----------
        correlation_threshold : float
            相関の閾値
        sync_threshold : float
            同期判定の閾値
        max_lag : int
            最大ラグ（因果解析用）
        distance_cutoff : float
            原子間距離カットオフ（Å）
        """
        self.correlation_threshold = correlation_threshold
        self.sync_threshold = sync_threshold
        self.max_lag = max_lag
        self.distance_cutoff = distance_cutoff

        self.xp = cp if HAS_GPU else np
        logger.info(f"🔺 AtomicNetworkGPU initialized (GPU: {HAS_GPU})")

    def analyze_network(
        self,
        trajectory: np.ndarray,
        anomaly_atoms: List[int],
        residue_mapping: Dict[int, List[int]],
        start_frame: int,
        end_frame: int,
    ) -> AtomicNetworkResult:
        """
        原子ネットワーク解析（メイン）

        Parameters
        ----------
        trajectory : np.ndarray
            軌道データ (n_frames, n_atoms, 3)
        anomaly_atoms : List[int]
            異常原子のリスト
        residue_mapping : Dict[int, List[int]]
            残基→原子のマッピング
        """
        if not anomaly_atoms:
            return AtomicNetworkResult()

        n_frames = end_frame - start_frame + 1

        # 単一フレームの場合
        if n_frames <= 1:
            return self._analyze_instantaneous_network(
                trajectory[start_frame], anomaly_atoms, residue_mapping
            )

        # 複数フレームの高度な解析
        return self._analyze_temporal_network(
            trajectory[start_frame : end_frame + 1], anomaly_atoms, residue_mapping
        )

    def _analyze_instantaneous_network(
        self, frame: np.ndarray, atoms: List[int], residue_mapping: Dict
    ) -> AtomicNetworkResult:
        """単一フレームの瞬間的協調ネットワーク"""
        result = AtomicNetworkResult(network_pattern="instantaneous")

        # 原子座標
        coords = frame[atoms]
        n_atoms = len(atoms)

        # 距離行列
        if HAS_GPU:
            coords_gpu = cp.asarray(coords)
            dist_matrix = cp_cdist(coords_gpu, coords_gpu)
            dist_matrix = cp.asnumpy(dist_matrix)
        else:
            from scipy.spatial.distance import cdist

            dist_matrix = cdist(coords, coords)

        # 近接原子のみネットワーク化
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if dist_matrix[i, j] < self.distance_cutoff:
                    link = AtomicNetworkLink(
                        from_atom=atoms[i],
                        to_atom=atoms[j],
                        from_residue=self._get_residue_id(atoms[i], residue_mapping),
                        to_residue=self._get_residue_id(atoms[j], residue_mapping),
                        link_type="sync",
                        strength=1.0 / (1.0 + dist_matrix[i, j]),
                        distance=dist_matrix[i, j],
                    )
                    result.sync_network.append(link)

        # ハブ原子の特定
        degree_count = Counter()
        for link in result.sync_network:
            degree_count[link.from_atom] += 1
            degree_count[link.to_atom] += 1

        result.hub_atoms = [atom for atom, count in degree_count.most_common(5)]

        # 残基間ブリッジ
        result.residue_bridges = self._detect_bridges(result.sync_network)

        return result

    def _analyze_temporal_network(
        self, trajectory: np.ndarray, atoms: List[int], residue_mapping: Dict
    ) -> AtomicNetworkResult:
        """複数フレームの時間的ネットワーク解析"""
        result = AtomicNetworkResult()

        # GPU転送
        if HAS_GPU:
            traj_gpu = cp.asarray(trajectory[:, atoms])
        else:
            traj_gpu = trajectory[:, atoms]

        n_frames, n_atoms = traj_gpu.shape[0], len(atoms)

        # 1. 適応的窓の計算
        adaptive_windows = self._compute_adaptive_windows(traj_gpu, atoms)
        result.adaptive_windows = adaptive_windows

        # 2. 相関行列の計算
        correlations = self._compute_correlations(traj_gpu, adaptive_windows)

        # 3. ネットワーク構築
        networks = self._build_networks(
            correlations, atoms, trajectory[0], residue_mapping
        )

        result.sync_network = networks["sync"]
        result.causal_network = networks["causal"]
        result.async_network = networks["async"]

        # 4. パターン識別
        result.network_pattern = self._identify_pattern(networks)

        # 5. ハブ原子の特定
        all_links = networks["sync"] + networks["causal"] + networks["async"]
        degree_count = Counter()
        for link in all_links:
            degree_count[link.from_atom] += 1
            degree_count[link.to_atom] += 1

        result.hub_atoms = [atom for atom, count in degree_count.most_common(10)]

        # 6. 残基間ブリッジ検出
        result.residue_bridges = self._detect_bridges(networks["causal"])

        return result

    def _compute_adaptive_windows(
        self, traj_gpu: Any, atoms: List[int]
    ) -> Dict[int, int]:
        """原子ごとの適応的窓サイズ計算"""
        n_frames = traj_gpu.shape[0]
        windows = {}

        for i, atom in enumerate(atoms):
            # 変位の変動性を計算
            if HAS_GPU:
                displacements = cp.diff(traj_gpu[:, i], axis=0)
                volatility = float(cp.std(cp.linalg.norm(displacements, axis=1)))
            else:
                displacements = np.diff(traj_gpu[:, i], axis=0)
                volatility = float(np.std(np.linalg.norm(displacements, axis=1)))

            # 変動性に基づく窓サイズ
            if volatility > 2.0:
                windows[atom] = min(5, n_frames)
            elif volatility < 0.5:
                windows[atom] = min(20, n_frames)
            else:
                windows[atom] = min(10, n_frames)

        return windows

    def _compute_correlations(self, traj_gpu: Any, windows: Dict[int, int]) -> Dict:
        """全原子ペアの相関計算（同期・非同期・因果）"""
        n_frames, n_atoms = traj_gpu.shape[:2]

        correlations = {
            "sync": np.zeros((n_atoms, n_atoms)),
            "lagged": np.zeros((n_atoms, n_atoms, self.max_lag + 1)),
            "async_strength": np.zeros((n_atoms, n_atoms)),
        }

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                window = min(
                    windows[list(windows.keys())[i]], windows[list(windows.keys())[j]]
                )

                # 時系列データ
                if HAS_GPU:
                    ts_i = cp.linalg.norm(traj_gpu[:, i], axis=1)
                    ts_j = cp.linalg.norm(traj_gpu[:, j], axis=1)
                else:
                    ts_i = np.linalg.norm(traj_gpu[:, i], axis=1)
                    ts_j = np.linalg.norm(traj_gpu[:, j], axis=1)

                # 同期相関
                if HAS_GPU:
                    sync_corr = float(cp.corrcoef(ts_i[:window], ts_j[:window])[0, 1])
                else:
                    sync_corr = np.corrcoef(ts_i[:window], ts_j[:window])[0, 1]

                correlations["sync"][i, j] = sync_corr
                correlations["sync"][j, i] = sync_corr

                # ラグ付き相関
                for lag in range(1, min(self.max_lag + 1, window)):
                    if HAS_GPU:
                        lagged_corr = float(cp.corrcoef(ts_i[:-lag], ts_j[lag:])[0, 1])
                    else:
                        lagged_corr = np.corrcoef(ts_i[:-lag], ts_j[lag:])[0, 1]

                    correlations["lagged"][i, j, lag] = lagged_corr

                # 非同期強度
                if abs(sync_corr) < 0.2:  # 同期してない
                    # 距離ベースの結合強度
                    if HAS_GPU:
                        mean_dist = float(
                            cp.mean(
                                cp.linalg.norm(traj_gpu[:, i] - traj_gpu[:, j], axis=1)
                            )
                        )
                    else:
                        mean_dist = np.mean(
                            np.linalg.norm(traj_gpu[:, i] - traj_gpu[:, j], axis=1)
                        )

                    if mean_dist < self.distance_cutoff:
                        correlations["async_strength"][i, j] = 1.0 / (1.0 + mean_dist)

        return correlations

    def _build_networks(
        self,
        correlations: Dict,
        atoms: List[int],
        first_frame: np.ndarray,
        residue_mapping: Dict,
    ) -> Dict:
        """ネットワーク構築"""
        networks = {"sync": [], "causal": [], "async": []}
        n_atoms = len(atoms)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                atom_i, atom_j = atoms[i], atoms[j]
                res_i = self._get_residue_id(atom_i, residue_mapping)
                res_j = self._get_residue_id(atom_j, residue_mapping)

                # 距離
                dist = np.linalg.norm(first_frame[atom_i] - first_frame[atom_j])

                # 同期ネットワーク
                sync_corr = correlations["sync"][i, j]
                if abs(sync_corr) > self.sync_threshold:
                    networks["sync"].append(
                        AtomicNetworkLink(
                            from_atom=atom_i,
                            to_atom=atom_j,
                            from_residue=res_i,
                            to_residue=res_j,
                            link_type="sync",
                            strength=abs(sync_corr),
                            distance=dist,
                        )
                    )

                # 因果ネットワーク
                max_lag_corr = 0.0
                best_lag = 0
                for lag in range(1, self.max_lag + 1):
                    lag_corr = correlations["lagged"][i, j, lag]
                    if abs(lag_corr) > abs(max_lag_corr):
                        max_lag_corr = lag_corr
                        best_lag = lag

                if abs(max_lag_corr) > self.correlation_threshold:
                    networks["causal"].append(
                        AtomicNetworkLink(
                            from_atom=atom_i,
                            to_atom=atom_j,
                            from_residue=res_i,
                            to_residue=res_j,
                            link_type="causal",
                            strength=abs(max_lag_corr),
                            lag=best_lag,
                            distance=dist,
                        )
                    )

                # 非同期ネットワーク
                async_strength = correlations["async_strength"][i, j]
                if async_strength > 0.3:
                    networks["async"].append(
                        AtomicNetworkLink(
                            from_atom=atom_i,
                            to_atom=atom_j,
                            from_residue=res_i,
                            to_residue=res_j,
                            link_type="async",
                            strength=async_strength,
                            distance=dist,
                        )
                    )

        return networks

    def _identify_pattern(self, networks: Dict) -> str:
        """ネットワークパターンの識別"""
        n_sync = len(networks["sync"])
        n_causal = len(networks["causal"])
        n_async = len(networks["async"])

        if n_sync > n_causal * 2:
            return "parallel"  # 同期的協調
        elif n_causal > n_sync * 2:
            return "cascade"  # カスケード伝播
        else:
            return "mixed"

    def _detect_bridges(self, links: List[AtomicNetworkLink]) -> List[ResidueBridge]:
        """残基間ブリッジの検出"""
        bridges_dict = defaultdict(list)

        for link in links:
            if link.from_residue != link.to_residue:
                key = (link.from_residue, link.to_residue)
                bridges_dict[key].append((link.from_atom, link.to_atom, link.strength))

        bridges = []
        for (from_res, to_res), atom_pairs in bridges_dict.items():
            total_strength = sum(s for _, _, s in atom_pairs)
            bridges.append(
                ResidueBridge(
                    from_residue=from_res,
                    to_residue=to_res,
                    bridge_atoms=[(a1, a2) for a1, a2, _ in atom_pairs],
                    total_strength=total_strength,
                    dominant_type="causal" if links else "sync",
                )
            )

        return sorted(bridges, key=lambda b: b.total_strength, reverse=True)

    def _get_residue_id(self, atom_id: int, mapping: Dict) -> int:
        """原子IDから残基IDを取得"""
        for res_id, atom_list in mapping.items():
            if atom_id in atom_list:
                return res_id
        return -1  # 不明


# ============================================
# Third Impact Analyzer v3.0
# ============================================


class ThirdImpactAnalyzer:
    """
    Third Impact Analytics v3.0
    原子レベル量子痕跡 + GPU高速ネットワーク解析
    """

    def __init__(
        self,
        residue_mapping: Optional[Dict[int, List[int]]] = None,
        sigma_threshold: float = 3.0,
        use_network_analysis: bool = True,
        use_gpu: bool = True,
    ):
        """
        Parameters
        ----------
        residue_mapping : Dict[int, List[int]], optional
            残基ID -> 原子IDリストのマッピング
        sigma_threshold : float
            統計的有意性の閾値
        use_network_analysis : bool
            ネットワーク解析を使用するか
        use_gpu : bool
            GPU加速を使用するか
        """
        self.residue_mapping = residue_mapping
        self.sigma_threshold = sigma_threshold
        self.use_network_analysis = use_network_analysis

        # ネットワーク解析器
        if use_network_analysis and use_gpu and HAS_GPU:
            self.atomic_network = AtomicNetworkGPU()
            logger.info("🚀 GPU-accelerated atomic network analysis enabled")
        elif use_network_analysis:
            self.atomic_network = AtomicNetworkGPU()  # GPU無しでも動く
            logger.info("🔺 CPU atomic network analysis enabled")
        else:
            self.atomic_network = None

        logger.info("🔺 Third Impact Analyzer v3.0 initialized")

    def analyze_critical_residues(
        self,
        lambda_result: Any,
        two_stage_result: Any,
        trajectory: np.ndarray,
        top_n: int = 3,
        **kwargs,
    ) -> Dict[str, ThirdImpactResult]:
        """
        異常残基の原子レベル解析（v3.0）
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔺 THIRD IMPACT v3.0 - Atomic Network Analysis")
        logger.info("=" * 60)

        start_time = time.time()
        results = {}

        # Top N残基の特定
        top_residues = self._identify_top_residues(two_stage_result, top_n)
        logger.info(f"Analyzing top {len(top_residues)} residues")

        # 各イベントの解析
        for event_name, residue_analysis in two_stage_result.residue_analyses.items():
            for residue_event in residue_analysis.residue_events:
                if residue_event.residue_id not in top_residues:
                    continue

                residue_id = residue_event.residue_id
                start_frame = residue_event.start_frame
                end_frame = residue_event.end_frame
                duration = end_frame - start_frame

                # イベントタイプの判定
                if duration <= 1:
                    event_type = "instantaneous"
                else:
                    event_type = "propagation"

                logger.info(f"\n📍 {event_name} - Residue {residue_id} ({event_type})")

                # 残基の原子を取得
                residue_atoms = self._get_residue_atoms(residue_id, trajectory.shape[1])

                # 基本解析
                if event_type == "instantaneous":
                    result = self._analyze_instantaneous_event(
                        residue_id,
                        event_name,
                        trajectory,
                        lambda_result.lambda_structures,
                        start_frame,
                        residue_atoms,
                    )
                else:
                    result = self._analyze_propagation_event(
                        residue_id,
                        event_name,
                        trajectory,
                        lambda_result.lambda_structures,
                        start_frame,
                        end_frame,
                        residue_atoms,
                    )

                # ネットワーク解析（NEW!）
                if self.atomic_network and result.quantum_atoms:
                    logger.info("   🌐 Analyzing atomic network...")
                    network_result = self.atomic_network.analyze_network(
                        trajectory=trajectory,
                        anomaly_atoms=list(result.quantum_atoms.keys()),
                        residue_mapping=self.residue_mapping,
                        start_frame=start_frame,
                        end_frame=end_frame,
                    )

                    result.atomic_network = network_result
                    result.n_network_links = (
                        len(network_result.sync_network)
                        + len(network_result.causal_network)
                        + len(network_result.async_network)
                    )
                    result.n_residue_bridges = len(network_result.residue_bridges)

                    # ネットワーク情報を原子痕跡に反映
                    self._update_quantum_traces_with_network(result)

                # 創薬ターゲット特定
                result.drug_target_atoms = self._identify_drug_targets(result)

                # ブリッジ原子もターゲットに（NEW!）
                if result.atomic_network and result.atomic_network.residue_bridges:
                    for bridge in result.atomic_network.residue_bridges[:3]:
                        for atom_pair in bridge.bridge_atoms[:2]:
                            result.bridge_target_atoms.extend(atom_pair)

                results[f"{event_name}_res{residue_id}"] = result

        computation_time = time.time() - start_time
        logger.info(f"\n🔺 Analysis complete in {computation_time:.2f}s")

        self._print_summary(results)
        return results

    def _analyze_instantaneous_event(
        self,
        residue_id: int,
        event_name: str,
        trajectory: np.ndarray,
        lambda_structures: Dict,
        frame: int,
        residue_atoms: List[int],
    ) -> ThirdImpactResult:
        """単一フレームイベントの解析"""
        result = ThirdImpactResult(
            event_name=event_name, residue_id=residue_id, event_type="instantaneous"
        )

        if frame == 0:
            return result

        # 全原子の変位を計算
        displacements = trajectory[frame] - trajectory[frame - 1]
        distances = np.linalg.norm(displacements, axis=1)

        # 統計量
        mean_d = np.mean(distances)
        std_d = np.std(distances)
        threshold = mean_d + self.sigma_threshold * std_d

        result.origin.mean_displacement = mean_d
        result.origin.std_displacement = std_d
        result.origin.threshold_used = threshold

        # 残基内で異常な原子を特定
        for atom_id in residue_atoms:
            if atom_id >= len(distances):
                continue

            z_score = (distances[atom_id] - mean_d) / (std_d + 1e-10)

            if z_score > self.sigma_threshold:
                trace = AtomicQuantumTrace(
                    atom_id=atom_id, residue_id=residue_id, displacement_zscore=z_score
                )

                # Lambda変化
                if "lambda_F_mag" in lambda_structures:
                    lambda_change = (
                        lambda_structures["lambda_F_mag"][frame]
                        - lambda_structures["lambda_F_mag"][frame - 1]
                    )
                    trace.lambda_change = float(lambda_change)

                # シグネチャー分類
                trace.quantum_signature = self._classify_signature(
                    z_score, trace.lambda_change
                )
                trace.confidence = min(z_score / 5.0, 1.0)

                result.quantum_atoms[atom_id] = trace
                result.origin.genesis_atoms.append(atom_id)

        # 統計更新
        result.n_quantum_atoms = len(result.quantum_atoms)
        if result.quantum_atoms:
            confidences = [t.confidence for t in result.quantum_atoms.values()]
            result.max_confidence = max(confidences)

            signatures = [t.quantum_signature for t in result.quantum_atoms.values()]
            if signatures:
                result.strongest_signature = Counter(signatures).most_common(1)[0][0]

        return result

    def _analyze_propagation_event(
        self,
        residue_id: int,
        event_name: str,
        trajectory: np.ndarray,
        lambda_structures: Dict,
        start_frame: int,
        end_frame: int,
        residue_atoms: List[int],
    ) -> ThirdImpactResult:
        """複数フレームイベントの解析（ネットワーク解析付き）"""
        result = ThirdImpactResult(
            event_name=event_name, residue_id=residue_id, event_type="propagation"
        )

        if start_frame == 0:
            return result

        # 起源原子の特定
        genesis_result = self._analyze_instantaneous_event(
            residue_id,
            event_name,
            trajectory,
            lambda_structures,
            start_frame,
            residue_atoms,
        )

        result.origin.genesis_atoms = genesis_result.origin.genesis_atoms
        result.quantum_atoms = genesis_result.quantum_atoms

        # 第一波の追跡（簡易版）
        max_propagation_frames = min(2, end_frame - start_frame)

        for delta in range(1, max_propagation_frames + 1):
            frame = start_frame + delta
            if frame >= len(trajectory):
                break

            wave_result = self._analyze_instantaneous_event(
                residue_id,
                f"{event_name}_wave{delta}",
                trajectory,
                lambda_structures,
                frame,
                residue_atoms,
            )

            new_atoms = [
                a
                for a in wave_result.origin.genesis_atoms
                if a not in result.origin.genesis_atoms
            ]
            result.origin.first_wave_atoms.extend(new_atoms)

            # 新規異常原子も追加
            for atom_id, trace in wave_result.quantum_atoms.items():
                if atom_id not in result.quantum_atoms:
                    result.quantum_atoms[atom_id] = trace

        # 統計更新
        result.n_quantum_atoms = len(result.quantum_atoms)
        if result.quantum_atoms:
            confidences = [t.confidence for t in result.quantum_atoms.values()]
            result.max_confidence = max(confidences)

            signatures = [t.quantum_signature for t in result.quantum_atoms.values()]
            if signatures:
                result.strongest_signature = Counter(signatures).most_common(1)[0][0]

        return result

    def _update_quantum_traces_with_network(self, result: ThirdImpactResult):
        """ネットワーク情報で量子痕跡を更新"""
        if not result.atomic_network:
            return

        # 接続度を計算
        degree_count = Counter()
        all_links = (
            result.atomic_network.sync_network
            + result.atomic_network.causal_network
            + result.atomic_network.async_network
        )

        for link in all_links:
            degree_count[link.from_atom] += 1
            degree_count[link.to_atom] += 1

        # ハブ原子をマーク
        hub_atoms = set(result.atomic_network.hub_atoms)

        # ブリッジ原子をマーク
        bridge_atoms = set()
        for bridge in result.atomic_network.residue_bridges:
            for atom_pair in bridge.bridge_atoms:
                bridge_atoms.update(atom_pair)

        # 量子痕跡を更新
        for atom_id, trace in result.quantum_atoms.items():
            trace.connectivity_degree = degree_count.get(atom_id, 0)
            trace.is_hub = atom_id in hub_atoms
            trace.is_bridge = atom_id in bridge_atoms

            # ネットワーク起源に追加
            if trace.is_hub:
                result.origin.network_initiators.append(atom_id)

    def _classify_signature(self, z_score: float, lambda_change: float) -> str:
        """量子シグネチャーの分類"""
        if z_score > 5.0:
            return "quantum_jump"
        elif z_score > 4.0 and abs(lambda_change) > 0.1:
            return "tunneling"
        elif z_score > 3.5:
            return "entanglement"
        elif z_score > 3.0:
            return "quantum_anomaly"
        else:
            return "thermal"

    def _identify_drug_targets(self, result: ThirdImpactResult) -> List[int]:
        """創薬ターゲット原子の特定（ネットワーク情報も考慮）"""
        targets = []

        # ハブ原子を最優先
        for atom_id, trace in result.quantum_atoms.items():
            if trace.is_hub and trace.confidence > 0.7:
                targets.append(atom_id)

        # ブリッジ原子も重要
        for atom_id, trace in result.quantum_atoms.items():
            if trace.is_bridge and atom_id not in targets:
                targets.append(atom_id)

        # 高信頼度の起源原子
        for atom_id, trace in result.quantum_atoms.items():
            if trace.confidence > 0.8 and atom_id not in targets:
                targets.append(atom_id)

        return targets[:7]  # 最大7個

    def _identify_top_residues(self, two_stage_result: Any, top_n: int) -> Set[int]:
        """上位異常残基を特定"""
        if not hasattr(two_stage_result, "global_residue_importance"):
            return set()

        importance_scores = two_stage_result.global_residue_importance
        sorted_residues = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        )

        return set(res_id for res_id, _ in sorted_residues[:top_n])

    def _get_residue_atoms(self, residue_id: int, n_atoms: int) -> List[int]:
        """残基に属する原子IDを取得"""
        if self.residue_mapping and residue_id in self.residue_mapping:
            return self.residue_mapping[residue_id]

        # フォールバック
        logger.warning(f"No mapping for residue {residue_id}, using fallback")
        atoms_per_residue = 15
        start_atom = residue_id * atoms_per_residue
        end_atom = min(start_atom + atoms_per_residue, n_atoms)
        return list(range(start_atom, end_atom))

    def _print_summary(self, results: Dict[str, ThirdImpactResult]):
        """解析サマリーの出力（v3.0）"""
        print("\n" + "=" * 60)
        print("🔺 THIRD IMPACT v3.0 SUMMARY")
        print("=" * 60)

        total_genesis = sum(len(r.origin.genesis_atoms) for r in results.values())
        total_quantum = sum(r.n_quantum_atoms for r in results.values())
        total_links = sum(r.n_network_links for r in results.values())
        total_bridges = sum(r.n_residue_bridges for r in results.values())

        print("\n📊 Statistics:")
        print(f"  - Events analyzed: {len(results)}")
        print(f"  - Genesis atoms: {total_genesis}")
        print(f"  - Quantum atoms: {total_quantum}")
        print(f"  - Network links: {total_links}")
        print(f"  - Residue bridges: {total_bridges}")

        # 各イベントの詳細
        for event_key, result in results.items():
            print(f"\n📍 {event_key} ({result.event_type}):")
            print(f"  - Residue: {result.residue_id}")
            print(f"  - Genesis atoms: {result.origin.genesis_atoms[:5]}")

            if result.atomic_network:
                print(f"  - Network pattern: {result.atomic_network.network_pattern}")
                print(f"  - Hub atoms: {result.atomic_network.hub_atoms[:3]}")

                if result.atomic_network.residue_bridges:
                    bridge = result.atomic_network.residue_bridges[0]
                    print(
                        f"  - Main bridge: Res{bridge.from_residue}→Res{bridge.to_residue}"
                    )
                    print(f"    Bridge atoms: {bridge.bridge_atoms[:2]}")

            if result.drug_target_atoms:
                print(f"  - Drug targets: {result.drug_target_atoms[:5]}")

            if result.bridge_target_atoms:
                print(f"  - Bridge targets: {result.bridge_target_atoms[:3]}")


# ============================================
# Integration Functions
# ============================================


def run_third_impact_analysis(
    lambda_result: Any,
    two_stage_result: Any,
    trajectory: np.ndarray,
    residue_mapping: Optional[Dict[int, List[int]]] = None,
    output_dir: Optional[Path] = None,
    use_network_analysis: bool = True,
    **kwargs,
) -> Dict[str, ThirdImpactResult]:
    """
    Third Impact解析の実行（v3.0）

    Parameters
    ----------
    use_network_analysis : bool
        原子ネットワーク解析を使用するか
    """
    logger.info("🔺 Starting Third Impact Analysis v3.0...")

    # アナライザー初期化
    analyzer = ThirdImpactAnalyzer(
        residue_mapping=residue_mapping,
        sigma_threshold=kwargs.get("sigma_threshold", 3.0),
        use_network_analysis=use_network_analysis,
        use_gpu=kwargs.get("use_gpu", True),
    )

    # 解析実行
    results = analyzer.analyze_critical_residues(
        lambda_result=lambda_result,
        two_stage_result=two_stage_result,
        trajectory=trajectory,
        top_n=kwargs.get("top_n", 3),
    )

    # 結果保存
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # JSON保存
        save_results_json(results, output_path)

        # レポート生成
        report = generate_impact_report(results)
        with open(output_path / "third_impact_v3_report.txt", "w") as f:
            f.write(report)

        # ネットワーク可視化用データ
        if use_network_analysis:
            save_network_data(results, output_path)

        logger.info(f"Results saved to {output_path}")

    return results


def save_results_json(results: Dict[str, ThirdImpactResult], output_path: Path):
    """結果をJSON形式で保存（v3.0）"""
    json_data = {}

    for event_key, result in results.items():
        json_data[event_key] = {
            "event_name": result.event_name,
            "residue_id": result.residue_id,
            "event_type": result.event_type,
            "n_quantum_atoms": result.n_quantum_atoms,
            "n_network_links": result.n_network_links,
            "n_residue_bridges": result.n_residue_bridges,
            "genesis_atoms": result.origin.genesis_atoms,
            "network_initiators": result.origin.network_initiators,
            "strongest_signature": result.strongest_signature,
            "max_confidence": float(result.max_confidence),
            "drug_target_atoms": result.drug_target_atoms,
            "bridge_target_atoms": result.bridge_target_atoms,
            "statistics": {
                "mean_displacement": float(result.origin.mean_displacement),
                "std_displacement": float(result.origin.std_displacement),
                "threshold": float(result.origin.threshold_used),
            },
        }

        # ネットワーク情報
        if result.atomic_network:
            json_data[event_key]["network"] = {
                "pattern": result.atomic_network.network_pattern,
                "hub_atoms": result.atomic_network.hub_atoms,
                "n_sync_links": len(result.atomic_network.sync_network),
                "n_causal_links": len(result.atomic_network.causal_network),
                "n_async_links": len(result.atomic_network.async_network),
                "n_bridges": len(result.atomic_network.residue_bridges),
            }

    with open(output_path / "third_impact_v3.json", "w") as f:
        json.dump(json_data, f, indent=2)


def save_network_data(results: Dict[str, ThirdImpactResult], output_path: Path):
    """ネットワーク可視化用データを保存"""
    for event_key, result in results.items():
        if not result.atomic_network:
            continue

        # NetworkX用エッジリスト
        edges = []

        for link in result.atomic_network.sync_network:
            edges.append(
                {
                    "source": link.from_atom,
                    "target": link.to_atom,
                    "type": "sync",
                    "weight": link.strength,
                }
            )

        for link in result.atomic_network.causal_network:
            edges.append(
                {
                    "source": link.from_atom,
                    "target": link.to_atom,
                    "type": "causal",
                    "weight": link.strength,
                    "lag": link.lag,
                }
            )

        # 保存
        network_file = output_path / f"{event_key}_network.json"
        with open(network_file, "w") as f:
            json.dump(
                {"edges": edges, "hubs": result.atomic_network.hub_atoms}, f, indent=2
            )


def generate_impact_report(results: Dict[str, ThirdImpactResult]) -> str:
    """レポート生成（v3.0）"""
    report = """
================================================================================
🔺 THIRD IMPACT ANALYSIS v3.0 - GPU Atomic Network Edition
================================================================================

"""

    # 統計サマリー
    total_genesis = sum(len(r.origin.genesis_atoms) for r in results.values())
    total_quantum = sum(r.n_quantum_atoms for r in results.values())
    total_links = sum(r.n_network_links for r in results.values())
    total_bridges = sum(r.n_residue_bridges for r in results.values())

    report += f"""EXECUTIVE SUMMARY
-----------------
Events Analyzed: {len(results)}
Genesis Atoms Identified: {total_genesis}
Quantum Atoms Detected: {total_quantum}
Network Links Discovered: {total_links}
Residue Bridges Found: {total_bridges}

DETAILED ANALYSIS
-----------------
"""

    for event_key, result in results.items():
        report += f"\n{event_key} ({result.event_type})\n"
        report += "=" * len(event_key) + "\n"
        report += f"Target Residue: {result.residue_id}\n"
        report += f"Genesis Atoms: {result.origin.genesis_atoms[:10]}\n"

        if result.origin.network_initiators:
            report += f"Network Hubs: {result.origin.network_initiators[:5]}\n"

        if result.atomic_network:
            report += "\nNetwork Analysis:\n"
            report += f"  Pattern: {result.atomic_network.network_pattern}\n"
            report += f"  Sync Links: {len(result.atomic_network.sync_network)}\n"
            report += f"  Causal Links: {len(result.atomic_network.causal_network)}\n"
            report += f"  Async Links: {len(result.atomic_network.async_network)}\n"

            if result.atomic_network.residue_bridges:
                report += "\nResidue Bridges:\n"
                for bridge in result.atomic_network.residue_bridges[:3]:
                    report += f"  Res{bridge.from_residue} → Res{bridge.to_residue} "
                    report += f"(strength: {bridge.total_strength:.3f})\n"
                    report += f"    Atoms: {bridge.bridge_atoms[:2]}\n"

        report += f"\nQuantum Signature: {result.strongest_signature}\n"
        report += f"Max Confidence: {result.max_confidence:.3f}\n"

        if result.drug_target_atoms:
            report += f"\nDrug Target Atoms: {result.drug_target_atoms}\n"

        if result.bridge_target_atoms:
            report += f"Bridge Target Atoms: {result.bridge_target_atoms[:5]}\n"

        report += "\nStatistics:\n"
        report += f"  μ_displacement: {result.origin.mean_displacement:.3f} Å\n"
        report += f"  σ_displacement: {result.origin.std_displacement:.3f} Å\n"
        report += f"  Detection threshold: {result.origin.threshold_used:.3f} Å\n"

    report += """
================================================================================
Generated by Third Impact Analytics v3.0 - GPU Atomic Network Edition
Powered by residue_network GPU technology adapted to atomic level
================================================================================
"""

    return report
