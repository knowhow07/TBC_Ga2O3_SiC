# Data Availability

This repository contains processed datasets, trained potential files, simulation inputs, simulation outputs, and post-processing scripts used for the Ga2O3/SiC TBC study.

## Included Data

The processed NEP training, validation, and test data are included in:

```text
nep_training/data/train.xyz
nep_training/data/val.xyz
nep_training/data/tests/*.xyz
nep_training/data/manifest.csv
```

The included split contains:

- 21065 training structures
- 5266 validation structures
- 2925 test structures
- 29256 total structures

The trained NEP model is included in:

```text
nep_training/output/nep.txt
```

Simulation inputs, outputs, and derived analysis data are included under:

```text
DFT/
EMD/
NEMD/
```

## Raw DFT/AIMD Provenance

The processed `xyz` files are included, but the full raw DFT/AIMD provenance tree used to generate every training structure is not fully self-contained in this folder. The file:

```text
nep_training/data/manifest.csv
```

contains redacted historical source paths under `<PATH_TO_DATA>/qe_jobs/...`. These paths should be treated as provenance records from the author's local computing environment unless the corresponding raw data are separately released.

## Restricted Files

VASP `POTCAR` files are not expected in the public release tree. VASP source code, binaries, pseudopotentials, and license-server information are excluded from public redistribution.

Obvious VASP-generated output and charge-density files have been moved to `_REVIEW_BEFORE_PUBLIC_RELEASE/` for human review. Public release should review VASP `OUTCAR`, `vasprun.xml`, `CHGCAR*`, `AECCAR*`, `XDATCAR`, `WAVECAR`, `PROCAR`, `DOSCAR`, `CHGDIFF.vasp`, and related outputs against the applicable VASP license and company/institutional rules before including them in any archive.

Remaining VASP-style structure files such as `*.vasp` should also be checked for provenance and source-license compatibility.

## Recommended Public Archive

For final publication, deposit the cleaned repository in a DOI-granting archive such as Zenodo, Figshare, Materials Cloud, or an institutional repository. Record the DOI here after release.

DOI: `TBD`

Repository URL: `TBD`
