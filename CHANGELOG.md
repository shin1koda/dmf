# Changelog

All notable changes to this project will be documented in this file.

## [1.2.2] - 2026-07-15

### Added
- Added `samples/sample_parallel.py` .

### Changed
- Some minor changes for JOSS revision.


## [1.2.1] - 2026-05-28

### Removed
- `parallel='mpi'` option from `DirectMaxFlux`, as the existing thread-based parallelization provides equivalent image-level parallelism.

## [1.2.0] - 2026-05-22

### Added
- Added `src/dmf/torch` for GPU acceleration, contributed by @t-0hmura .

## [1.1.1] - 2026-05-21

### Added
- Added `CHANGELOG.md` (this file).
- Added `mpi4py` under optional dependencies.
- Added `.gitignore` files to `tests/` and `sample/` directories.

### Changed
- Updated `requires-python` to `>=3.10`.
- Updated `README.md` to include applications using PyDMF
