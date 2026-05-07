# Release Review Summary

Review date: 2026-04-30

This file records the public-release preparation actions taken in this working tree. It is not a legal approval.

## Files Edited

Top-level release and licensing documents edited:

- `LICENSE.md`
- `README.md`
- `DATA_AVAILABILITY.md`
- `REPRODUCIBILITY.md`
- `THIRD_PARTY_NOTICES.md`
- `PUBLIC_RELEASE_CHECKLIST.md`
- `RELEASE_REVIEW_SUMMARY.md`

Mechanical path redaction was also applied across text files containing local absolute paths. Original local workspace, home-directory, and shared-storage prefixes containing a private username were replaced with `<PATH_TO_DATA>`. After this pass, 282 text files contain the neutral placeholder `<PATH_TO_DATA>`, including scripts, logs, environment snapshots, and `nep_training/data/manifest.csv`.

## Files Flagged

The following files remain in the active tree and require human provenance/license review before public release because they use VASP-style structure extensions or may be externally sourced:

- `DFT/phonon/Ga2O3/POSCAR_pre.vasp`
- `DFT/phonon/SiC/POSCAR_pre.vasp`
- `DFT/electron/m201/SiCH_dipole/PRIMCELL.vasp`
- `DFT/electron/100/SiCH_dipole_4H/PRIMCELL.vasp`
- `EMD/SiC_rotate/300k/SiC-mp-11714.vasp`

The following categories should also remain flagged until reviewed:

- VASP-derived files and raw electronic-structure outputs.
- Externally sourced structures, datasets, or figures.
- Archival conda environment snapshots with machine-specific provenance.
- Trained model and training data release approval.

## Files Moved To Quarantine

The following 72 files were moved into `_REVIEW_BEFORE_PUBLIC_RELEASE/` while preserving their original relative paths. They should be omitted from public archives unless legal and institutional review explicitly clears redistribution.

### `DFT/electron/m201/inter_dipole`

- `OUTCAR`
- `vasprun.xml`
- `CHGCAR_sum`
- `XDATCAR`
- `CHGCAR`
- `CHGCAR_SiC`
- `CHGCAR_Ga2O3`
- `CHGDIFF.vasp`
- `AECCAR2`
- `PROCAR`
- `CHGCAR_valence`
- `DOSCAR`
- `AECCAR1`
- `AECCAR0`

### `DFT/electron/m201`

- `CHGDIFF.vasp`

### `DFT/electron/m201/GaO_dipole`

- `OUTCAR`
- `vasprun.xml`
- `CHGCAR_sum`
- `XDATCAR`
- `CHGCAR`
- `AECCAR2`
- `PROCAR`
- `DOSCAR`
- `AECCAR1`
- `AECCAR0`

### `DFT/electron/m201/SiCH_dipole`

- `OUTCAR`
- `vasprun.xml`
- `CHGCAR_sum`
- `XDATCAR`
- `CHGCAR`
- `AECCAR2`
- `PROCAR`
- `DOSCAR`
- `AECCAR1`
- `AECCAR0`

### `DFT/electron/100/inter_dipole`

- `OUTCAR`
- `vasprun.xml`
- `CHGCAR_sum`
- `XDATCAR`
- `CHGCAR`
- `CHGCAR_SiC`
- `CHGCAR_Ga2O3`
- `CHGDIFF.vasp`
- `AECCAR2`
- `PROCAR`
- `CHGCAR_valence`
- `DOSCAR`
- `AECCAR1`
- `AECCAR0`

### `DFT/electron/100`

- `CHGDIFF.vasp`

### `DFT/electron/100/GaO_dipole`

- `OUTCAR`
- `vasprun.xml`
- `CHGCAR_sum`
- `XDATCAR`
- `CHGCAR`
- `AECCAR2`
- `PROCAR`
- `DOSCAR`
- `AECCAR1`
- `AECCAR0`

### `DFT/electron/100/SiCH_dipole_4H`

- `OUTCAR`
- `vasprun.xml`
- `CHGCAR_sum`
- `XDATCAR`
- `CHGCAR`
- `AECCAR2`
- `PROCAR`
- `DOSCAR`
- `AECCAR1`
- `AECCAR0`

### `DFT/ifc`

- `DFT/ifc/m201-106/script/vasprun.xml`
- `DFT/ifc/100-106/script/vasprun.xml`

## Unresolved Human Review Items

- Confirm legal/company approval for the Apache-2.0 and CC BY-NC 4.0 license split.
- Confirm whether any quarantined VASP-generated files can be redistributed. If not, keep `_REVIEW_BEFORE_PUBLIC_RELEASE/` out of the public release.
- Confirm that no VASP source, binaries, `POTCAR`, pseudopotentials, license-server addresses, or institution-restricted VASP materials are present in the public tree.
- Review the five active `*.vasp` structure files for provenance and redistribution terms.
- Confirm third-party notices for GPUMD, LAMMPS, Phonopy, phono3py, pymatgen, ASE, Bader tools, OVITO, VESTA, and Python dependencies.
- Confirm approval to release `nep_training/output/nep.txt`, NEP restart/checkpoint files, and the processed train/validation/test datasets.
- Decide whether redacted historical paths in `nep_training/data/manifest.csv` are acceptable as provenance records or should be replaced with relative public provenance paths.
- Update public metadata before release: `CITATION.cff`, DOI, repository URL, contact information, and any final manuscript/preprint file.
