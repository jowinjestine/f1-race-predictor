"""Execute 07_shap_analysis.ipynb locally, bypassing GCS calls.

Injects a temporary cell that patches storage functions to local-only,
runs the notebook, then removes the patch cell from the saved output.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import nbformat
from nbclient import NotebookClient

PATCH_SOURCE = '''
# --- LOCAL EXECUTION PATCH (auto-injected, will be removed) ---
import f1_predictor.data.storage as _storage
import pandas as _pd
from pathlib import Path as _P

def _local_only_load(gcs_blob, local_path):
    print(f"  [local] Reading {local_path}")
    return _pd.read_parquet(local_path)

def _local_only_sync(model_type, local_dir=None):
    local_dir = local_dir or _P("data/training")
    files = sorted(local_dir.glob(f"model_{model_type}_*.parquet"))
    print(f"  [local] Found {len(files)} files for Model {model_type}")
    return files

_storage.load_from_gcs_or_local = _local_only_load
_storage.sync_training_from_gcs = _local_only_sync
print("GCS calls patched to local-only.")
'''

PATCH_TAG = "LOCAL_EXEC_PATCH"

nb_path = Path("notebooks/07_shap_analysis.ipynb")
print(f"Loading {nb_path} ...")
nb = nbformat.read(nb_path, as_version=4)

patch_cell = nbformat.v4.new_code_cell(source=PATCH_SOURCE.strip())
patch_cell.metadata["tags"] = [PATCH_TAG]
nb.cells.insert(2, patch_cell)

print("Executing notebook (timeout=1200s) ...")
client = NotebookClient(nb, timeout=1200, kernel_name="python3")
client.execute()

nb.cells = [c for c in nb.cells if PATCH_TAG not in c.metadata.get("tags", [])]

nbformat.write(nb, nb_path)
print(f"\nDone — executed notebook saved to {nb_path}")
