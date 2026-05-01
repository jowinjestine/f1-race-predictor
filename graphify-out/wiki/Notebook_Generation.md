# Notebook Generation

## Overview
This community contains the script responsible for programmatically generating Jupyter training notebooks for all four F1 predictor models (A, B, C, D) plus a model comparison notebook. The script uses helper functions to create markdown and code cells, then assembles them into complete `.ipynb` files. This approach ensures reproducible, version-controlled notebook generation rather than manual editing.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| `make_training_notebooks.py` | Script | `scripts/make_training_notebooks.py` |
| `main()` | Function | `scripts/make_training_notebooks.py` |
| `make_notebook()` | Function | `scripts/make_training_notebooks.py` |
| `make_cell()` | Function | `scripts/make_training_notebooks.py` |
| `md()` | Function | `scripts/make_training_notebooks.py` |
| `code()` | Function | `scripts/make_training_notebooks.py` |
| `make_model_a()` | Function | `scripts/make_training_notebooks.py` |
| `make_model_b()` | Function | `scripts/make_training_notebooks.py` |
| `make_model_c()` | Function | `scripts/make_training_notebooks.py` |
| `make_model_d()` | Function | `scripts/make_training_notebooks.py` |
| `make_model_comparison()` | Function | `scripts/make_training_notebooks.py` |

## Relationships

### Internal
- `main()` --calls--> `make_notebook()` [1.0]
- `main()` --calls--> `make_model_a()` [1.0]
- `main()` --calls--> `make_model_b()` [1.0]
- `main()` --calls--> `make_model_c()` [1.0]
- `main()` --calls--> `make_model_d()` [1.0]
- `main()` --calls--> `make_model_comparison()` [1.0]
- `make_model_a()` --calls--> `md()`, `code()` [1.0]
- `make_model_b()` --calls--> `md()`, `code()` [1.0]
- `make_model_c()` --calls--> `md()`, `code()` [1.0]
- `make_model_d()` --calls--> `md()`, `code()` [1.0]
- `make_model_comparison()` --calls--> `md()`, `code()` [1.0]
- `md()` --calls--> `make_cell()` [1.0]
- `code()` --calls--> `make_cell()` [1.0]

### Cross-community
- Generated notebooks reference feature modules from [Lap Feature Engineering](Lap_Feature_Engineering.md) (Models A/B)
- Generated notebooks reference feature modules from [Race Feature Engineering](Race_Feature_Engineering.md) (Model C)
- Generated notebooks use the [GCS Storage Layer](GCS_Storage_Layer.md) for data I/O
- Model training connects to the [CV Splits & Init](CV_Splits_Init.md) for cross-validation setup
- Generated notebooks are part of the [Feature Build Functions](Feature_Build_Functions.md) pipeline

## Source Files
- `scripts/make_training_notebooks.py`
