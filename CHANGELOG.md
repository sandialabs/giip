# Changelog

## [0.2.0] - 2022-02-01
### Added
- Two-dimensional rotation matrix generation
- Factored out species-agnostic weight generation function
- Example01, the two-dimensional defective metal
### Bugfixes
- Single-weight short-circuit code in distance condensation function
- Removed reference to float32, rendering weight calculation float-size agnostic

## [0.1.0] - 2022-01-19
### Added
- Gaussian Integral Inner Product (GIIP) for single or multiple weight sets
- Three weight set calculators
- GIIP distance for single or multiple weight sets
- GIIP distance condensation from multiple weight sets
- Global (exhaustive) and local (refined) orientation optimization for neighborhood registration
- Rotation matrix sets from Peter Mahler Larsen's hyperspherical covering project, at 1,2,3,4, and 5 degree resolution