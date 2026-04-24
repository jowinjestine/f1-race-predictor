## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current

## GPU Training

- GPU detection: `src/f1_predictor/models/gpu.py` — supports NVIDIA CUDA and AMD ROCm
- XGBoost GPU requires CUDA (forced to CPU on AMD ROCm)
- LightGBM GPU uses OpenCL (works on both NVIDIA and AMD)
- PyTorch uses HIP on ROCm — `torch.cuda.*` API works identically
- DL models (GRU, FT-Transformer, MLP): `src/f1_predictor/models/architectures.py`
- DL utilities (training loop, early stopping): `src/f1_predictor/models/dl_utils.py`
- WSL2 setup: `bash scripts/setup_wsl_env.sh`
- Local training: `bash scripts/run_training_wsl.sh`
- GCE fallback: `bash scripts/run_training_remote.sh` (preserved, unchanged)
