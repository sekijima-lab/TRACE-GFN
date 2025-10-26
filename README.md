# TRACE-GFN: Transformer for Reaction-Aware Compound Exploration with GFlowNet in QSAR-Guided Molecular Design

**TRACE-GFN** is a generative flow network (GFlowNet) framework for designing drug-like molecules through interpretable chemical reaction pathways. Please refer to the paper for more detailed information.


## Installation

### Quick Install

```bash
bash install.sh
source .venv/bin/activate
```

### Manual Installation

```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate

# Install PyTorch Geometric dependencies (CUDA 12.1)
uv pip install torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

### For CPU-only Installation

If you don't have CUDA available, modify the PyTorch installation in `pyproject.toml` to use CPU-only versions.

## Download Model Parameters

Please download trained weights for the Transformer from [Figshare here](https://figshare.com/articles/software/Weights_of_conditional_unconditional_Transformer/25853551), and place the weights as `Transformer.pth` in the `src/gflownet/models/ckpts/Transformer/` directory.

Then, the directory substructure is as follows:

```
models/
└── ckpts/
    ├── GCN/
    │    └── GCN.pth
    └── Transformer/
         └── Transformer.pth
```

## Usage

### Basic Usage

Generate molecules optimized for DRD2 binding starting from a specific compound:

```bash
python -u src/gflownet/tasks/qsar_reactions.py \
  --protein_name "DRD2" \
  --init_compound_idx 1 \
  --condition 16.0 \
  --max_depth 5
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--protein_name` | Target protein: "DRD2", "AKT1", or "CXCR4" | Required |
| `--init_compound_idx` | Index of starting material | Required |
| `--condition` | Temperature parameter (higher = more exploration) | 16.0 |
| `--max_depth` | Maximum number of reaction steps | 5 |

### Starting Materials

Starting materials are specified in SMILES format in:
```
src/gflownet/data/{PROTEIN_NAME}/init_compound_{IDX}.smi
```

For example, `src/gflownet/data/DRD2/init_compound_1.smi` contains:
```
OC1CCc2cc(F)ccc21
```

You can create custom starting materials by adding new `.smi` files with your desired SMILES strings.

### Example Commands

**DRD2 optimization with high exploration:**
```bash
python -u src/gflownet/tasks/qsar_reactions.py \
  --protein_name "DRD2" \
  --init_compound_idx 1 \
  --condition 32.0 \
  --max_depth 7
```

**AKT1 optimization with conservative exploration:**
```bash
python -u src/gflownet/tasks/qsar_reactions.py \
  --protein_name "AKT1" \
  --init_compound_idx 6 \
  --condition 16.0 \
  --max_depth 5
```

**CXCR4 optimization using probabilistic sampling:**
```bash
python -u src/gflownet/tasks/qsar_reactions.py \
  --protein_name "CXCR4" \
  --init_compound_idx 11 \
  --condition 16.0 \
  --max_depth 6 \
```

## Output

Training outputs are saved to:
```
./logs/{PROTEIN_NAME}_reactions_{TIMESTAMP}/
```

### Monitoring Training

The training progress is logged to Weights & Biases (wandb) under the project `{PROTEIN_NAME}_TRACER-GFN`. Metrics include:

- Rewards (binding affinity predictions)
- Loss values (trajectory balance, GCN, Transformer)
- Sampling diversity (unique molecule rate)
- Training time and throughput

## Configuration

Key hyperparameters can be modified in [src/gflownet/config.py](src/gflownet/config.py) or via command-line arguments.


## Supported Protein Targets

TRACE-GFN includes pre-trained QSAR models for three protein targets:

1. **DRD2**
   - Relevant for: Antipsychotics, Parkinson's disease treatments
   - QSAR model: `src/gflownet/models/qsar_DRD2_optimized.pkl`

2. **AKT1** 
   - Relevant for: Cancer therapies, metabolic disorders
   - QSAR model: `src/gflownet/models/qsar_AKT1_optimized.pkl`

3. **CXCR4**
   - Relevant for: HIV treatments, cancer metastasis inhibitors
   - QSAR model: `src/gflownet/models/qsar_CXCR4_optimized.pkl`

### Adding Custom Targets

To add a new protein target:

1. Train a QSAR model (e.g., using Morgan fingerprints and Random Forest)
2. Save the model as `src/gflownet/models/qsar_{PROTEIN_NAME}_optimized.pkl`
3. Update the protein name options in [src/gflownet/tasks/qsar_reactions.py](src/gflownet/tasks/qsar_reactions.py)
4. Prepare starting materials in `src/gflownet/data/{PROTEIN_NAME}/init_compound_*.smi`

## Reaction Templates

TRACE-GFN uses reaction templates derived from the USPTO dataset:

- **Template library**: `src/gflownet/data/label_template.json` (1000 templates)
- **Training data**: `src/gflownet/data/USPTO/` (tokenized reaction examples)

Reaction templates are represented as SMARTS patterns that define molecular transformations. The GCN learns to predict which templates are applicable to each molecule based on structural features.


## Project Structure

```
TRACE-GFN/
├── src/gflownet/
│   ├── tasks/
│   │   └── qsar_reactions.py       # Main entry point
│   ├── models/
│   │   ├── GCN/                    # Graph convolution network
│   │   ├── Transformer/            # Product generation model
│   │   ├── mlp.py                  # Partition function predictor
│   │   └── qsar_*.pkl              # Pre-trained QSAR models
│   ├── algo/
│   │   ├── trajectory_balance_synthesis.py  # Training objective
│   │   └── reaction_sampling.py    # Trajectory generation
│   ├── data/
│   │   ├── DRD2/, AKT1/, CXCR4/   # Protein-specific data
│   │   ├── USPTO/                  # Reaction templates
│   │   └── sampling_iterator.py    # Data loading
│   ├── trainer.py                  # Base trainer class
│   ├── online_trainer.py           # Online training implementation
│   └── config.py                   # Configuration dataclasses
├── install.sh                      # Installation script
└── pyproject.toml                  # Dependencies
```

