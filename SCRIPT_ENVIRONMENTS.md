# Script Environments

This folder uses two conda environments for post-processing and analysis scripts.

## Base Analysis Environment

Most Python scripts were run in the base conda environment. For reproducibility, use the minimal environment file:

```bash
conda env create -f environment.yml
conda activate tbc-base
```

Use this environment for:

- `nep_training/*.py`
- `EMD/**/*.py`
- `NEMD/**/*.py`
- `other/figure/**/*.py`

## DFT / Phoebe Environment

DFT-related scripts were run in the `phoebe` conda environment. For reproducibility, use:

```bash
conda env create -f environment_phoebe.yml
conda activate phoebe
```

Use this environment for:

- `DFT/phonon/**/*.py`
- `DFT/ifc/**/*.py`
- `DFT/electron/**/*.py`

## External Programs

The conda files cover Python packages only. Full repetition of the simulations also requires separately installed external programs, including VASP, GPUMD, LAMMPS, Phonopy command-line tools, and Bader analysis tools where applicable.

VASP and POTCAR-related files are subject to institutional license restrictions and are not made reproducible solely by these conda files.
