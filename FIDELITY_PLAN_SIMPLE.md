# Fidelity Plan (Simple)

## Goal
Integrate a projection fidelity (DPS-style) term into SDS training, validate geometry alignment, and reduce memory usage.

## Checklist
- [x] Add fidelity projection operator using `grid_sample` with rays
- [x] Use `VESDEGuidance.train_step_with_Fidelity` for SDS + fidelity
- [x] Add chunking for slices/rays to control memory
- [x] Build a notebook to compare `A(image_gt)` vs. `projs`
- [x] Validate data ranges and normalization

## Notes
- Use `fidelity_weight` to scale the projection gradient term.
- Use `fidelity_res`, `fidelity_slice_chunk`, and `fidelity_rays_chunk` to reduce OOM risk.
- Verify geometry by comparing forward projections with stored `projs`.
