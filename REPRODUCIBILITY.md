# Reproducibility Guide

This guide summarizes how to reproduce the main results from existing files.

For a coverage audit, see:

```text
reproducibility_audit_report.md
```

## 1. Create Environments

Base analysis:

```bash
conda env create -f environment.yml
conda activate tbc-base
```

DFT/phonon analysis:

```bash
conda env create -f environment_phoebe.yml
conda activate phoebe
```

## 2. NEP Training and Validation Outputs

```bash
cd <PATH_TO_DATA>/open_access/nep_training
python plot_loss.py
python plot_test_nep.py
```

Relevant files:

```text
nep.in
data/train.xyz
data/val.xyz
data/tests/*.xyz
output/nep.txt
output/loss.out
tests/rmse_summary.csv
```

## 3. Bulk Thermal Conductivity

Ga2O3:

```bash
cd <PATH_TO_DATA>/open_access/EMD/Ga2O3
python nep_exp.py
```

SiC:

```bash
cd <PATH_TO_DATA>/open_access/EMD/SiC_rotate
python nep_exp.py
```

## 4. NEMD TBC Trends

Length dependence:

```bash
cd <PATH_TO_DATA>/open_access/NEMD/length
python compare_tbc.py
```

Temperature dependence:

```bash
cd <PATH_TO_DATA>/open_access/NEMD/temperature
python plot_temp_tbc.py
```

## 5. Phonon Validation

Use the `phoebe` environment:

```bash
conda activate phoebe
```

SiC:

```bash
cd <PATH_TO_DATA>/open_access/DFT/phonon/SiC
python compare_nep.py
```

Ga2O3:

```bash
cd <PATH_TO_DATA>/open_access/DFT/phonon/Ga2O3
python compare_nep.py
```

## 6. Charge and Interface Metrics

Use scripts under:

```text
DFT/electron/100
DFT/electron/m201
DFT/electron/*/inter_dipole
DFT/ifc/100-106
DFT/ifc/m201-106
```

These workflows depend on VASP outputs, Bader outputs, Phonopy files, and Python packages in `environment_phoebe.yml`.

Some raw VASP output and charge-density files needed for full charge-density reproduction have been quarantined under `_REVIEW_BEFORE_PUBLIC_RELEASE/`. They are not public-release approved unless legal and institutional review clears redistribution.

## 7. Spectral and Coherence Analyses

Use scripts and results under:

```text
NEMD/analysis/100/shc
NEMD/analysis/100/pdos
NEMD/analysis/m201/shc
NEMD/analysis/m201/pdos
```

The scripts are directory-sensitive. Run each script from the folder where it is located unless the script says otherwise.

## 8. Full Simulation Repetition

Full repetition requires external MD/DFT executables and substantial compute:

- VASP for DFT/AIMD and electronic-structure workflows;
- GPUMD for NEP and some MD/spectral workflows;
- LAMMPS for EMD workflows;
- Phonopy/phono3py for phonon and IFC workflows;
- Bader tools for charge analysis.

The manuscript reports that a typical 4 ns NEMD simulation with about 25k atoms takes about 10 to 15 GPU hours on H100-class hardware.
