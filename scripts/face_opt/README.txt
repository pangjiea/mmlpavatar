SMPL-X face-keypoint multi-view optimization (head-focused)

Modules:
- datasets.py: Load cameras/images from a subject directory and pick cameras with visible faces.
- mp_face.py: MediaPipe-based face detection and landmark extraction (2D pixels).
- smplx_utils.py: SMPL-X loading, pose composition, 3D face landmarks projection.
- optim_head.py: Multi-view L2 loss and parameter optimization (fix betas).
- run_opt.py: CLI entrypoint orchestrating the pipeline and saving debug outputs.

Runtime outputs are written to output/face_opt/{subject}/{frame:06d}/.
Includes: selected_cameras.json, per-camera landmark overlays, and optimized npz.

Environment: see env_micromamba.sh to create an isolated env without touching system Python.
