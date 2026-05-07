# Third-Party Software and License Notes

This project relies on external scientific software. Users are responsible for installing and licensing these tools where applicable. Third-party materials are governed by their own licenses and terms; nothing in this repository relicenses third-party software, proprietary inputs, proprietary outputs, or externally sourced data.

| Software | Role in this project | License / terms |
|---|---|---|
| VASP | DFT/AIMD calculations and charge-density/electronic-structure outputs | Proprietary; requires a valid VASP license |
| GPUMD | NEP training and GPUMD-based MD/transport workflows | GPLv3 |
| LAMMPS | Molecular dynamics simulations | GPLv2 |
| Phonopy | Phonon and force-constant analysis | BSD 3-Clause |
| phono3py | Phonon/thermal-transport support | BSD 3-Clause |
| pymatgen | Structure and charge-density processing | MIT |
| ASE | Structure conversion and preprocessing | LGPL |
| NumPy/SciPy/pandas/matplotlib/h5py/PyYAML/openpyxl | Python analysis stack | See each package's upstream license |

## VASP Files

VASP `POTCAR` files, pseudopotential files, source code, binaries, and license-server information are not included for public redistribution and are excluded from this repository's licenses.

VASP-generated or VASP-derived outputs may be subject to VASP license terms and company/institutional redistribution rules. Files such as `OUTCAR`, `vasprun.xml`, `WAVECAR`, `CHGCAR`, `AECCAR*`, `XDATCAR`, `PROCAR`, `DOSCAR`, charge-density difference files, and similar raw electronic-structure outputs must be reviewed before public release. Files moved to `_REVIEW_BEFORE_PUBLIC_RELEASE/` are quarantined and should not be included in a public archive unless explicitly cleared.

Structure files with VASP-style extensions, such as `*.vasp`, are not automatically proprietary, but their provenance should be checked before release.

## Repository License

Original code and scripts are licensed under Apache-2.0. Original processed data, generated figures, documentation, and manuscript-supporting materials are licensed under CC BY-NC 4.0 unless otherwise noted. See `LICENSE.md`.
