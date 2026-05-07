# Physics-Grounded Understanding of Thermal Boundary Conductance from a Feedforward Neural Network Potential

This repository contains the data, trained neural network potential, simulation inputs/outputs, and analysis scripts associated with the manuscript:

**Physics-Grounded Understanding of Thermal Boundary Conductance from a Feedforward Neural Network Potential**

The study develops a unified Ga-O-Si-C-H neuroevolution potential (NEP) for Ga2O3/SiC interfacial heat transport and uses it to analyze thermal boundary conductance (TBC) trends with respect to transport length, temperature, and interface orientation.

## Public Release Status

This folder is close to a reproducibility archive, but it should still be reviewed before being uploaded as a public repository.

Important:

- VASP-related inputs and outputs and other externally licensed files should still be reviewed against the applicable license terms before public distribution.
- High-risk VASP-derived output files have been moved to `_REVIEW_BEFORE_PUBLIC_RELEASE/` and should not be included in a public archive unless cleared.
- Original code and scripts are licensed under Apache-2.0.
- Original processed data, generated figures, and documentation are licensed under CC BY-NC 4.0 unless otherwise noted.
- See `PUBLIC_RELEASE_CHECKLIST.md` before uploading this repository.

## Repository Layout

```text
.
├── DFT/                     # DFT, phonon, IFC, charge-density, and binding analyses
├── EMD/                     # Equilibrium MD bulk thermal-conductivity simulations
├── NEMD/                    # Nonequilibrium MD TBC and mechanistic analyses
├── nep_training/            # NEP training, model, train/val/test data, and validation plots
├── other/figure/NN/         # NEP architecture/workflow drawing scripts and images
├── _REVIEW_BEFORE_PUBLIC_RELEASE/
│                            # Quarantined restricted/uncertain files; omit unless cleared
├── TBC_preprint.pdf   # To be added
├── environment.yml          # Minimal base analysis environment
├── environment_phoebe.yml   # Minimal DFT/phonon analysis environment
└── SCRIPT_ENVIRONMENTS.md   # Which environment to use for each script family
```

## Scientific Scope

The repository supports these tasks:

- training and validating a unified Ga-O-Si-C-H NEP;
- reproducing loss curves and energy/force/virial validation plots;
- comparing NEP and DFT phonon dispersions for SiC and beta-Ga2O3;
- reproducing bulk thermal-conductivity trends from EMD/HNEMD-style analyses;
- reproducing NEMD TBC trends with length, temperature, and interface orientation;
- analyzing spectral heat current, PDOS, coherence, covariance-based frequency-bin coupling, charge redistribution, binding energy, and cross-interface harmonic IFC metrics.

## Environments

Two conda environments are provided.

### Base Analysis Environment

Use for most post-processing scripts in `nep_training`, `EMD`, `NEMD`, and `other/figure`:

```bash
conda env create -f environment.yml
conda activate tbc-base
```

### DFT / Phoebe Environment

Use for DFT, phonon, charge-density, and IFC analysis scripts:

```bash
conda env create -f environment_phoebe.yml
conda activate phoebe
```

The files `environment_from_history.yml` and `environment_from_history_phoebe.yml`, if present, are archival snapshots of the author's conda environments. They are useful provenance records but are not recommended as the primary public installation files because they include many machine-specific transitive dependencies.

## External Software Requirements

The conda files only cover Python packages. Repeating the full workflow also requires separately installed simulation and analysis programs:

| Software | Role |
|---|---|
| VASP | DFT/AIMD, charge-density, and reference electronic-structure calculations |
| GPUMD | NEP training and GPUMD-based MD/spectral workflows |
| LAMMPS | EMD simulations used for bulk thermal conductivity |
| Phonopy | Phonon and force-constant workflows |
| phono3py | Optional phonon/thermal-transport related support |
| Bader analysis code | Charge partitioning |
| OVITO / VESTA | Visualization of structures and volumetric data |

You must install and license these programs separately where required.

## Data Included

The processed NEP dataset is included:

- `nep_training/data/train.xyz`: 21065 structures
- `nep_training/data/val.xyz`: 5266 structures
- `nep_training/data/tests/*.xyz`: 2925 test structures
- `nep_training/data/manifest.csv`: 29256 data rows plus header

These counts match the manuscript's reported split of 21065 / 5266 / 2925 train / validation / test structures.

The trained model is:

```text
nep_training/output/nep.txt
```

The main NEP input is:

```text
nep_training/nep.in
```

## Quick Start: Existing Outputs

Most manuscript-level analyses can be inspected from existing outputs without rerunning expensive simulations.

Examples:

```bash
cd <PATH_TO_DATA>/open_access
conda activate tbc-base
```

NEP loss and parity plots:

```bash
cd nep_training
python plot_loss.py
python plot_test_nep.py
```

TBC versus length:

```bash
cd <PATH_TO_DATA>/open_access/NEMD/length
python compare_tbc.py
```

TBC versus temperature:

```bash
cd <PATH_TO_DATA>/open_access/NEMD/temperature
python plot_temp_tbc.py
```

Bulk thermal conductivity comparison:

```bash
cd <PATH_TO_DATA>/open_access/EMD/Ga2O3
python nep_exp.py

cd <PATH_TO_DATA>/open_access/EMD/SiC_rotate
python nep_exp.py
```

DFT/phonon comparison examples:

```bash
conda activate phoebe

cd <PATH_TO_DATA>/open_access/DFT/phonon/SiC
python compare_nep.py

cd <PATH_TO_DATA>/open_access/DFT/phonon/Ga2O3
python compare_nep.py
```

Exact command details may vary because several scripts assume they are run from their own directory.

## Reproducing Main Manuscript Items

| Manuscript item | Main files/folders |
|---|---|
| Figure 1, workflow and NEP architecture | `other/figure/NN/plot_nn.py`, `other/figure/NN/plot_nn_simple.py`, generated `nep_nn_*` images |
| Figure 2a, training loss | `nep_training/output/loss.out`, `nep_training/plot_loss.py` |
| Figure 2b-c, parity plots | `nep_training/plot_test_nep.py`, `nep_training/tests/parity_*`, `nep_training/tests/rmse_summary.csv` |
| Figure 3a-b, phonon dispersions | `DFT/phonon/SiC`, `DFT/phonon/Ga2O3` |
| Figure 3c-d, bulk thermal conductivity | `EMD/Ga2O3`, `EMD/SiC_rotate` |
| Figure 4a, NEMD temperature profile | `NEMD/length/*/*/compute.out`, `NEMD/length/*/*/plot_temp.py` |
| Figure 4b, TBC vs length | `NEMD/length/compare_tbc.py`, `NEMD/length/TBC_compare_100_vs_201.txt` |
| Figure 4c, TBC vs temperature | `NEMD/temperature/plot_temp_tbc.py`, `NEMD/temperature/TBC_vs_temperature_100_vs_minus201.txt` |
| Figure 5, length mechanism | `NEMD/analysis/*/shc`, `NEMD/analysis/*/pdos`, `coherence_length.py`, `spec_*` |
| Figure 6, temperature mechanism | `NEMD/analysis/*/shc/results_temp`, `results_freq_bin_contrib`, `NEMD/analysis/*/pdos/results` |
| Figure 7a, charge-density difference | `DFT/electron/100`, `DFT/electron/m201`, `charge_diff.py`, `charge_diff_compare.py` |
| Figure 7b, orientation spectral conductance | `NEMD/analysis/*/shc/results_orien`, `spec_orien.py`, `bin_orien.py` |
| Table 1, interface metrics | `DFT/electron/*/inter_dipole`, `DFT/ifc/*`, `cross_ifc.py`, `formation.py`, `bond_length.py` |
| Figure 8, appendix coupling analysis | `NEMD/analysis/m201/pdos/results_matrix_compare`, `coupling_temp_diagonal.py` |

For a detailed audit of coverage and gaps, see `reproducibility_audit_report.md`.

## Expected Compute Cost

Post-processing from existing outputs is relatively lightweight and can usually run on a CPU workstation.

Full simulation repetition is expensive. The manuscript reports that a typical 4 ns NEMD simulation with about 25k atoms takes roughly 10 to 15 GPU hours on H100-class hardware. Full repetition of all independent runs, lengths, temperatures, orientations, and spectral analyses requires substantial GPU time.

## Known Limitations of This Release

This archive is not yet a fully automated reproduction package.

Known limitations:

- Several scripts rely on current working directory and local folder naming conventions.
- Some scripts select the newest log/result file by timestamp.
- Exact multi-panel paper figure assembly scripts are not centralized.
- Raw DFT/AIMD provenance for every NEP training structure is not fully self-contained; `manifest.csv` records redacted historical source paths under `<PATH_TO_DATA>/qe_jobs/...`.
- Proprietary or uncertain VASP files and outputs require legal review before public redistribution. Quarantined files under `_REVIEW_BEFORE_PUBLIC_RELEASE/` are not release-approved.

## Recommended Public-Release Cleanup

Before public upload:

1. Confirm whether quarantined VASP `OUTCAR`, `vasprun.xml`, `CHGCAR*`, `AECCAR*`, `XDATCAR`, `PROCAR`, `DOSCAR`, `CHGDIFF.vasp`, and related outputs may be redistributed under the applicable institutional license.
2. Confirm company approval for Apache-2.0 on original code/scripts and CC BY-NC 4.0 on original data, figures, and documentation.

## Citation

If you use this repository, cite the associated manuscript. A draft citation metadata file is provided in `CITATION.cff`; update authors, DOI, and URL after publication.

## Contact

Update this section before public release with the corresponding author, lab, or issue tracker URL.
