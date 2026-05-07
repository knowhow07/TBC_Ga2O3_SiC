# Contributing

This repository is primarily a research reproducibility archive.

## Reporting Issues

When reporting a reproducibility issue, include:

- the script or command that failed;
- the working directory used to run the command;
- the conda environment name;
- the Python version;
- relevant traceback or log output;
- whether external tools such as VASP, GPUMD, LAMMPS, Phonopy, or Bader were available.

## Making Changes

Keep changes scoped and reproducible:

- avoid changing raw data files unless correcting a documented error;
- preserve original simulation outputs where possible;
- put new post-processing outputs in clearly named files;
- document any new dependencies in `environment.yml` or `environment_phoebe.yml`;
- update `README.md` and figure/table mappings when adding or moving scripts.

## Large Files

Large simulation outputs should be added only when they are necessary to reproduce a manuscript result. For public release, consider using a DOI-granting data archive for large files and keeping only checksums or download instructions in the repository.
