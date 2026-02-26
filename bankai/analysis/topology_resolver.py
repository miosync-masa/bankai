"""
Topology Resolver - PDB原子名解決モジュール
============================================

Third Impact v3.0の出力を人間可読な原子名に変換する。
PDBファイルのATOMレコードをパースして:
  atom_id 134 → "TRP9-CA"
  atom_id 136 → "TRP9-CB" 
のように解決する。
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("bankai.analysis.topology_resolver")


@dataclass
class AtomInfo:
    """原子の完全な情報"""
    atom_id: int          # 0-indexed（trajectory内のindex）
    atom_serial: int      # PDB serial number（1-indexed）
    atom_name: str        # "CA", "CB", "CZ", "OH", "NE1" etc.
    residue_name: str     # "TYR", "TRP", "GLU" etc.
    residue_seq: int      # PDB残基番号（1-indexed）
    residue_id: int       # 0-indexed（bankai内部ID）
    chain_id: str = "A"
    element: str = ""

    @property
    def full_name(self) -> str:
        """完全な原子名: TYR2-CZ"""
        return f"{self.residue_name}{self.residue_seq}-{self.atom_name.strip()}"

    @property
    def short_name(self) -> str:
        """短い名前: Y2-CZ (1文字アミノ酸コード)"""
        code = THREE_TO_ONE.get(self.residue_name, "X")
        return f"{code}{self.residue_seq}-{self.atom_name.strip()}"


# 3文字→1文字変換テーブル
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # 非標準
    "HIE": "H", "HID": "H", "HIP": "H", "CYX": "C",
    "ACE": "X", "NME": "X",  # キャップ
}


class TopologyResolver:
    """
    PDBファイルから原子名を解決するクラス。
    
    Usage:
        resolver = TopologyResolver.from_pdb("5awl.pdb")
        name = resolver.resolve(134)  # → "TRP9-CA"
        names = resolver.resolve_list([134, 136, 138])  # → ["TRP9-CA", "TRP9-CB", "TRP9-OG1"]
    """
    
    def __init__(self):
        self.atoms: dict[int, AtomInfo] = {}  # atom_id → AtomInfo
        self.residue_names: dict[int, str] = {}  # residue_id → "TYR1"
        self._loaded = False
    
    @classmethod
    def from_pdb(cls, pdb_path: str, protein_indices: Optional[list[int]] = None) -> "TopologyResolver":
        """
        PDBファイルからTopologyResolverを構築。
        
        Parameters
        ----------
        pdb_path : str
            PDBファイルパス
        protein_indices : list[int], optional
            タンパク質原子のインデックス（前処理でフィルタされた場合）
            None の場合はPDBのATOM行を順番に使う
        """
        resolver = cls()
        resolver._parse_pdb(pdb_path, protein_indices)
        return resolver
    
    @classmethod
    def from_mapping(cls, mapping_path: str, pdb_path: str, 
                     protein_indices: Optional[list[int]] = None) -> "TopologyResolver":
        """
        residue_atom_mapping.json + PDBから構築（より正確）。
        
        mapping_path の residue_id と PDB の残基番号を照合し、
        trajectory のインデックスと原子名を正確に対応づける。
        """
        resolver = cls()
        resolver._parse_pdb(pdb_path, protein_indices)
        
        # マッピングファイルで残基IDを補正
        if Path(mapping_path).exists():
            with open(mapping_path) as f:
                mapping = json.load(f)
            resolver._apply_mapping_correction(mapping)
        
        return resolver

    def _parse_pdb(self, pdb_path: str, protein_indices: Optional[list[int]] = None):
        """PDBファイルをパース"""
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            logger.warning(f"PDB file not found: {pdb_path}")
            return
        
        # PDBのATOM行を収集
        pdb_atoms = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        atom_serial = int(line[6:11].strip())
                        atom_name = line[12:16].strip()
                        residue_name = line[17:20].strip()
                        chain_id = line[21].strip() or "A"
                        residue_seq = int(line[22:26].strip())
                        element = line[76:78].strip() if len(line) > 76 else ""
                        
                        pdb_atoms.append({
                            "serial": atom_serial,
                            "name": atom_name,
                            "res_name": residue_name,
                            "chain": chain_id,
                            "res_seq": residue_seq,
                            "element": element,
                        })
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Skipping PDB line: {line.strip()} ({e})")
                        continue
        
        if not pdb_atoms:
            logger.warning("No ATOM records found in PDB")
            return
        
        logger.info(f"📖 Parsed {len(pdb_atoms)} atoms from PDB")
        
        # protein_indices がある場合: trajectory のインデックスとPDB原子を対応
        if protein_indices is not None:
            # protein_indices[i] = PDB内の原子index (0-based)
            # trajectory のatom_id i → PDB原子 protein_indices[i]
            for traj_idx, pdb_idx in enumerate(protein_indices):
                if pdb_idx < len(pdb_atoms):
                    pdb_atom = pdb_atoms[pdb_idx]
                    self._add_atom(traj_idx, pdb_atom)
        else:
            # protein_indices がない場合: PDBのATOM行を順番に対応
            for traj_idx, pdb_atom in enumerate(pdb_atoms):
                self._add_atom(traj_idx, pdb_atom)
        
        # 残基名マッピング構築
        self._build_residue_names()
        self._loaded = True
        
        logger.info(f"✅ TopologyResolver loaded: {len(self.atoms)} atoms, "
                    f"{len(self.residue_names)} residues")
    
    def _add_atom(self, traj_idx: int, pdb_atom: dict):
        """原子情報を追加"""
        # 残基IDの決定（0-indexed）
        # 最初に見つかった残基を0とする
        if not hasattr(self, '_res_seq_to_id'):
            self._res_seq_to_id = {}
            self._next_res_id = 0
        
        res_key = (pdb_atom["chain"], pdb_atom["res_seq"])
        if res_key not in self._res_seq_to_id:
            self._res_seq_to_id[res_key] = self._next_res_id
            self._next_res_id += 1
        
        res_id = self._res_seq_to_id[res_key]
        
        self.atoms[traj_idx] = AtomInfo(
            atom_id=traj_idx,
            atom_serial=pdb_atom["serial"],
            atom_name=pdb_atom["name"],
            residue_name=pdb_atom["res_name"],
            residue_seq=pdb_atom["res_seq"],
            residue_id=res_id,
            chain_id=pdb_atom["chain"],
            element=pdb_atom["element"],
        )
    
    def _build_residue_names(self):
        """残基名マッピングを構築"""
        for atom_info in self.atoms.values():
            if atom_info.residue_id not in self.residue_names:
                self.residue_names[atom_info.residue_id] = (
                    f"{atom_info.residue_name}{atom_info.residue_seq}"
                )
    
    def _apply_mapping_correction(self, mapping: dict):
        """residue_atom_mappingで残基IDを補正"""
        # mapping: {"0": {"atoms": [0,1,...], "n_atoms": 23}, ...}
        for res_id_str, info in mapping.items():
            res_id = int(res_id_str)
            for atom_id in info.get("atoms", []):
                if atom_id in self.atoms:
                    self.atoms[atom_id].residue_id = res_id
        
        # 残基名を再構築
        self.residue_names.clear()
        self._build_residue_names()
    
    # ========================================
    # Public API
    # ========================================
    
    def resolve(self, atom_id: int) -> str:
        """
        原子IDを人間可読な名前に解決。
        
        resolve(134) → "TRP9-CA"
        解決できない場合 → "atom_134"
        """
        if atom_id in self.atoms:
            return self.atoms[atom_id].full_name
        return f"atom_{atom_id}"
    
    def resolve_short(self, atom_id: int) -> str:
        """
        短い名前で解決。
        
        resolve_short(134) → "W9-CA"
        """
        if atom_id in self.atoms:
            return self.atoms[atom_id].short_name
        return f"a{atom_id}"
    
    def resolve_list(self, atom_ids: list[int], short: bool = False) -> list[str]:
        """原子IDリストを一括解決"""
        if short:
            return [self.resolve_short(a) for a in atom_ids]
        return [self.resolve(a) for a in atom_ids]
    
    def resolve_residue(self, residue_id: int) -> str:
        """
        残基IDを名前に解決。
        
        resolve_residue(0) → "TYR1"
        resolve_residue(8) → "TRP9"
        """
        return self.residue_names.get(residue_id, f"res_{residue_id}")
    
    def get_info(self, atom_id: int) -> Optional[AtomInfo]:
        """原子の完全な情報を取得"""
        return self.atoms.get(atom_id)
    
    def get_backbone_atoms(self, residue_id: int) -> list[int]:
        """残基の主鎖原子を取得"""
        backbone_names = {"N", "CA", "C", "O"}
        return [
            a.atom_id for a in self.atoms.values()
            if a.residue_id == residue_id and a.atom_name in backbone_names
        ]
    
    def get_sidechain_atoms(self, residue_id: int) -> list[int]:
        """残基の側鎖原子を取得"""
        backbone_names = {"N", "CA", "C", "O", "H", "HA"}
        return [
            a.atom_id for a in self.atoms.values()
            if a.residue_id == residue_id and a.atom_name not in backbone_names
        ]
    
    def is_loaded(self) -> bool:
        """トポロジーが読み込まれているか"""
        return self._loaded
    
    def summary(self) -> str:
        """サマリー文字列"""
        if not self._loaded:
            return "TopologyResolver: not loaded"
        
        lines = [f"TopologyResolver: {len(self.atoms)} atoms, {len(self.residue_names)} residues"]
        for res_id in sorted(self.residue_names.keys()):
            res_name = self.residue_names[res_id]
            n_atoms = sum(1 for a in self.atoms.values() if a.residue_id == res_id)
            lines.append(f"  {res_id}: {res_name} ({n_atoms} atoms)")
        return "\n".join(lines)


# ============================================
# Third Impact レポート名前解決関数
# ============================================

def resolve_report_text(report_text: str, resolver: TopologyResolver) -> str:
    """
    Third Impactレポートのテキストを原子名解決済みに変換。
    
    Before: "Target Residue: 8\nGenesis Atoms: [134, 136, 138]"
    After:  "Target Residue: TRP9 (8)\nGenesis Atoms: [TRP9-CA(134), TRP9-CB(136), TRP9-OG1(138)]"
    """
    import re
    
    if not resolver.is_loaded():
        return report_text
    
    resolved = report_text
    
    # "Target Residue: N" → "Target Residue: NAME (N)"
    def replace_target_residue(match):
        res_id = int(match.group(1))
        name = resolver.resolve_residue(res_id)
        return f"Target Residue: {name} ({res_id})"
    resolved = re.sub(r'Target Residue: (\d+)', replace_target_residue, resolved)
    
    # "[N, N, N]" パターンの原子IDリストを解決
    def replace_atom_list(match):
        prefix = match.group(1)  # "Genesis Atoms: " etc
        ids_str = match.group(2)  # "134, 136, 138"
        
        try:
            atom_ids = [int(x.strip()) for x in ids_str.split(",")]
            resolved_names = [f"{resolver.resolve(a)}({a})" for a in atom_ids]
            return f"{prefix}[{', '.join(resolved_names)}]"
        except ValueError:
            return match.group(0)
    
    # Genesis Atoms, Network Hubs, Drug Target Atoms, Bridge Target Atoms
    for field_name in ["Genesis Atoms", "Network Hubs", "Drug Target Atoms", 
                  "Bridge Target Atoms", "Hub atoms", "Bridge atoms"]:
        pattern = rf'({field_name}: )\[([0-9, ]+)\]'
        resolved = re.sub(pattern, replace_atom_list, resolved)
    
    # "ResN→ResM" パターン
    def replace_res_bridge(match):
        from_id = int(match.group(1))
        to_id = int(match.group(2))
        from_name = resolver.resolve_residue(from_id)
        to_name = resolver.resolve_residue(to_id)
        return f"{from_name}({from_id})→{to_name}({to_id})"
    resolved = re.sub(r'Res(\d+)→Res(\d+)', replace_res_bridge, resolved)
    
    return resolved


# ============================================
# Enhanced Report Generator
# ============================================

def generate_resolved_report(results: dict, resolver: TopologyResolver) -> str:
    """
    名前解決済みのThird Impactレポートを生成。
    
    Before:
      Target Residue: 8
      Genesis Atoms: [134, 136, 138]
      Network Hubs: [134, 136, 138]
    
    After:
      Target Residue: TRP9 (8)
      Genesis Atoms: [TRP9-CA(134), TRP9-CB(136), TRP9-NE1(138)]
      Network Hubs: [TRP9-CA(134), TRP9-CB(136), TRP9-NE1(138)]
    """
    report = """
================================================================================
🔺 THIRD IMPACT ANALYSIS v3.1 - Topology-Resolved Edition
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

TOPOLOGY INFO
-----------------
{resolver.summary()}

DETAILED ANALYSIS
-----------------
"""
    
    for event_key, result in results.items():
        res_name = resolver.resolve_residue(result.residue_id)
        report += f"\n{event_key} ({result.event_type})\n"
        report += "=" * len(event_key) + "\n"
        report += f"Target Residue: {res_name} ({result.residue_id})\n"
        
        # Genesis Atoms - 名前解決！
        if result.origin.genesis_atoms:
            genesis_names = resolver.resolve_list(result.origin.genesis_atoms[:10])
            report += f"Genesis Atoms: {genesis_names}\n"
            
            # 主鎖 vs 側鎖の分類
            backbone = []
            sidechain = []
            for atom_id in result.origin.genesis_atoms:
                info = resolver.get_info(atom_id)
                if info:
                    if info.atom_name in {"N", "CA", "C", "O"}:
                        backbone.append(info.full_name)
                    else:
                        sidechain.append(info.full_name)
            
            if backbone:
                report += f"  ├ Backbone: {backbone}\n"
            if sidechain:
                report += f"  └ Sidechain: {sidechain}\n"
        else:
            report += "Genesis Atoms: []\n"
        
        # Network Hubs
        if result.origin.network_initiators:
            hub_names = resolver.resolve_list(result.origin.network_initiators[:5])
            report += f"Network Hubs: {hub_names}\n"
        
        # ネットワーク解析
        if result.atomic_network:
            report += "\nNetwork Analysis:\n"
            report += f"  Pattern: {result.atomic_network.network_pattern}\n"
            report += f"  Sync Links: {len(result.atomic_network.sync_network)}\n"
            report += f"  Causal Links: {len(result.atomic_network.causal_network)}\n"
            report += f"  Async Links: {len(result.atomic_network.async_network)}\n"
            
            # ハブ原子
            if result.atomic_network.hub_atoms:
                hub_resolved = resolver.resolve_list(result.atomic_network.hub_atoms[:5])
                report += f"  Hub Atoms: {hub_resolved}\n"
            
            # 残基間ブリッジ - 名前解決！
            if result.atomic_network.residue_bridges:
                report += "\nResidue Bridges:\n"
                for bridge in result.atomic_network.residue_bridges[:3]:
                    from_name = resolver.resolve_residue(bridge.from_residue)
                    to_name = resolver.resolve_residue(bridge.to_residue)
                    report += f"  {from_name} → {to_name} "
                    report += f"(strength: {bridge.total_strength:.3f})\n"
                    
                    for a1, a2 in bridge.bridge_atoms[:3]:
                        n1 = resolver.resolve(a1)
                        n2 = resolver.resolve(a2)
                        report += f"    {n1} ↔ {n2}\n"
        
        # Quantum Signature
        report += f"\nQuantum Signature: {result.strongest_signature}\n"
        report += f"Max Confidence: {result.max_confidence:.3f}\n"
        
        # Drug Targets - 名前解決！
        if result.drug_target_atoms:
            report += "\n🎯 Drug Target Atoms:\n"
            for atom_id in result.drug_target_atoms:
                info = resolver.get_info(atom_id)
                trace = result.quantum_atoms.get(atom_id)
                if info and trace:
                    report += f"  • {info.full_name} (atom {atom_id})"
                    report += f" — {trace.quantum_signature}"
                    report += f", conf={trace.confidence:.3f}"
                    if trace.is_hub:
                        report += " [HUB]"
                    if trace.is_bridge:
                        report += " [BRIDGE]"
                    report += "\n"
                elif info:
                    report += f"  • {info.full_name} (atom {atom_id})\n"
        
        # Bridge Targets
        if result.bridge_target_atoms:
            bridge_names = resolver.resolve_list(result.bridge_target_atoms[:5])
            report += f"Bridge Target Atoms: {bridge_names}\n"
        
        report += "\nStatistics:\n"
        report += f"  μ_displacement: {result.origin.mean_displacement:.3f} Å\n"
        report += f"  σ_displacement: {result.origin.std_displacement:.3f} Å\n"
        report += f"  Detection threshold: {result.origin.threshold_used:.3f} Å\n"
    
    report += """
================================================================================
Generated by Third Impact Analytics v3.1 - Topology-Resolved Edition
"Every atom has a name. Every name tells a story." — 環ちゃん 💕
================================================================================
"""
    
    return report


# ============================================
# JSON出力も名前解決対応
# ============================================

def save_resolved_json(results: dict, resolver: TopologyResolver, output_path: Path):
    """名前解決済みJSON"""
    json_data = {}
    
    for event_key, result in results.items():
        entry = {
            "event_name": result.event_name,
            "residue_id": result.residue_id,
            "residue_name": resolver.resolve_residue(result.residue_id),
            "event_type": result.event_type,
            "n_quantum_atoms": result.n_quantum_atoms,
            "n_network_links": result.n_network_links,
            "n_residue_bridges": result.n_residue_bridges,
            "max_confidence": float(result.max_confidence),
            "strongest_signature": result.strongest_signature,
        }
        
        # Genesis atoms with names
        entry["genesis_atoms"] = [
            {
                "atom_id": a,
                "atom_name": resolver.resolve(a),
                "is_sidechain": _is_sidechain(resolver.get_info(a)),
            }
            for a in result.origin.genesis_atoms
        ]
        
        # Drug targets with names
        entry["drug_targets"] = []
        for atom_id in result.drug_target_atoms:
            trace = result.quantum_atoms.get(atom_id)
            target = {
                "atom_id": atom_id,
                "atom_name": resolver.resolve(atom_id),
            }
            if trace:
                target["signature"] = trace.quantum_signature
                target["confidence"] = float(trace.confidence)
                target["is_hub"] = trace.is_hub
                target["is_bridge"] = trace.is_bridge
            entry["drug_targets"].append(target)
        
        # Network info
        if result.atomic_network:
            entry["network"] = {
                "pattern": result.atomic_network.network_pattern,
                "hub_atoms": [
                    {"atom_id": a, "name": resolver.resolve(a)}
                    for a in result.atomic_network.hub_atoms[:5]
                ],
                "n_sync": len(result.atomic_network.sync_network),
                "n_causal": len(result.atomic_network.causal_network),
                "n_async": len(result.atomic_network.async_network),
            }
            
            # Bridges
            if result.atomic_network.residue_bridges:
                entry["network"]["bridges"] = [
                    {
                        "from": resolver.resolve_residue(b.from_residue),
                        "to": resolver.resolve_residue(b.to_residue),
                        "strength": float(b.total_strength),
                        "atoms": [
                            {
                                "from": resolver.resolve(a1),
                                "to": resolver.resolve(a2),
                            }
                            for a1, a2 in b.bridge_atoms[:3]
                        ]
                    }
                    for b in result.atomic_network.residue_bridges[:5]
                ]
        
        json_data[event_key] = entry
    
    with open(output_path / "third_impact_v31_resolved.json", "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Resolved JSON saved: {output_path / 'third_impact_v31_resolved.json'}")


def _is_sidechain(info: Optional[AtomInfo]) -> bool:
    """側鎖原子かどうか"""
    if info is None:
        return False
    return info.atom_name not in {"N", "CA", "C", "O", "H", "HA"}


# ============================================
# Integration: run_quantum_validation_pipelineへの組み込み
# ============================================

def create_resolver_from_pipeline(
    topology_path: Optional[str] = None,
    protein_indices_path: Optional[str] = None,
    atom_mapping_path: Optional[str] = None,
) -> Optional[TopologyResolver]:
    """
    パイプラインの引数からTopologyResolverを構築。
    
    run_quantum_validation_pipeline() から呼ばれる想定:
    
    ```python
    # run_full_analysis.py 内
    resolver = create_resolver_from_pipeline(
        topology_path=topology_path,
        protein_indices_path=protein_indices_path,
        atom_mapping_path=atom_mapping_path,
    )
    ```
    """
    if topology_path is None or not Path(topology_path).exists():
        logger.info("No topology file provided, atom names will not be resolved")
        return None
    
    # protein_indices の読み込み
    protein_indices = None
    if protein_indices_path and Path(protein_indices_path).exists():
        import numpy as np
        protein_indices = np.load(protein_indices_path).tolist()
        logger.info(f"Loaded protein indices: {len(protein_indices)} atoms")
    
    # Resolver構築
    try:
        if atom_mapping_path and Path(atom_mapping_path).exists():
            resolver = TopologyResolver.from_mapping(
                mapping_path=atom_mapping_path,
                pdb_path=topology_path,
                protein_indices=protein_indices,
            )
        else:
            resolver = TopologyResolver.from_pdb(
                pdb_path=topology_path,
                protein_indices=protein_indices,
            )
        
        if resolver.is_loaded():
            logger.info(f"✅ TopologyResolver ready: {resolver.summary().split(chr(10))[0]}")
            return resolver
        else:
            logger.warning("TopologyResolver failed to load")
            return None
    except Exception as e:
        logger.warning(f"TopologyResolver error: {e}")
        return None
