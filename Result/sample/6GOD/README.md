# Structural Analysis Conditions

## Target Structure
- **PDB ID:** 6GOD  
- **Resolution:** 1.35 Å (ultra-high resolution)  
- **Mutation:** G12D (Glycine 12 → Aspartic acid)  
- **Native Binding State:** GDP-bound form with Mg<sup>2+</sup> ion  

---

# Actual Simulation Conditions (Important)

## Apo-State Analysis
The structure was analyzed in the **apo form** (ligand-free state).

### Excluded Components
- GDP ligand removed  
- Mg<sup>2+</sup> ion removed  

### Reason
Ligand parameter generation in GROMACS would significantly increase system preparation complexity.

### Implication
Structural stabilization effects induced by GDP binding are not represented in this simulation.

---

# Residue Range
- **Analyzed residues:** 1–167  

Although the native protein contains 169 residues, the C-terminal residues (Lys168 and Lys169) were incomplete in the crystal structure.

### Missing Atoms
- Side-chain atoms of Lys168 and Lys169 were absent.

### Impact on Analysis
The core structural region is fully preserved; therefore, the omission does not affect the G12D structural analysis.

---

# Solvation Conditions
- **Water model:** TIP3P  
- **Simulation box:** Cubic box with 1.0 nm padding from the protein surface  
- **Ion conditions:** Na<sup>+</sup>/Cl<sup>−</sup> ions added for charge neutralization  

---

# Molecular Dynamics Conditions
- **Force field:** AMBER99SB-ILDN  
- **Temperature:** 300 K (after NPT equilibration)  
- **Time step:** 0.01 ps (ultra-high temporal resolution)  
- **Total simulation time:** 1 ns  
- **Trajectory size:** 100,000 frames  

---

# Important Considerations for Interpretation

Because the simulation was performed in the GDP-free apo state, discussions regarding canonical active/inactive conformations are not applicable.

However, the simulation is still suitable for evaluating:

- Intrinsic structural instability caused by the G12D mutation  
- Local conformational distortions  
- Mutation-induced dynamic perturbations  

If abnormal structural behavior is observed even in the apo state, similar instability may also persist in the GDP-bound state.
