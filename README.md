# ICM Algorithm 1 - TruthfulQA Implementation

Implementation of Algorithm 1 (Internal Coherence Maximization) from ["Unsupervised Elicitation of Language Models"](https://arxiv.org/abs/2506.10139).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export HYPERBOLIC_API_KEY="your-key-here"

# Quick test (30 train, 20 test samples)
python run_icm.py --model meta-llama/Meta-Llama-3.1-405B \
  --train_samples 30 --test_samples 20 --max_icm_iters 100

# Full run (256 train, 100 test)
python run_icm.py --model meta-llama/Meta-Llama-3.1-405B

# View latest results
cat runs/latest/results.json
open runs/latest/four_bars.png
```

## What This Does

1. **Runs ICM on train set** → Generates labels via mutual predictability (Algorithm 1)
2. **Uses ICM labels** → Predicts test set with many-shot prompting
3. **Compares to baselines** → Random, Zero-shot, Golden labels
4. **Generates figure** → Four-bar graph (`outputs/four_bars.png`)

## Files

- `run_icm.py` - Main script (runs everything)
- `icm_algorithm.py` - ICM Algorithm 1 implementation
- `critique.md` - Part 2 critique
- `data/` - TruthfulQA train/test data
- `outputs/` - Results and graphs

## Algorithm

ICM finds labels that maximize:
```
U(D) = α * P_θ(D)
```

Where `P_θ(D)` is mutual predictability: how well each label predicts others.

**Search:** Simulated annealing with temperature cooling.

## Output Structure

Each run creates a timestamped directory with hyperparameters:
```
runs/
├── 20251106_120000_model-Meta-Llama-3.1-405B_k6_trainfull_testfull_iters500_seed42/
│   ├── results.json                  # Summary metrics
│   ├── test_predictions.json         # All test predictions
│   ├── icm_generated_labels.json     # Train labels from ICM
│   ├── four_bars.png                 # Comparison graph
│   └── run_metadata.json             # Run configuration
└── latest -> [symlink to most recent run]
```

**No overwrites** - each run is preserved with full hyperparameter tracking

**Graph shows:**
- Random baseline
- Zero-shot baseline
- ICM (ours)
- Golden labels (100%)

## Parameters

```
--model           Base model (default: Meta-Llama-3.1-405B)
--train_samples   Limit train examples (optional, for testing)
--test_samples    Limit test examples (optional, for testing)
--max_icm_iters   ICM iterations (default: 500)
--k               In-context examples (default: 6)
--seed            Random seed (default: 42)
--output_dir      Base output directory (default: runs)
```

**Output naming:** Automatic timestamped directories with all hyperparameters:
```
runs/YYYYMMDD_HHMMSS_model-{MODEL}_k{K}_train{N}_test{M}_iters{I}_seed{S}/
```

## Notes

- Uses **BASE model** (not instruct) for pure probabilities
- **No fine-tuning** - prompt-based only as per instructions
- **No consistency fix** - ignored per work test instructions
- Full run on 256 train + 100 test takes ~30-45 minutes
- Can submit while code is running (per work test rules)

## Cost Estimate

- Quick test (30/20): ~$2
- Full run (256/100): ~$15-20

## Citation

```
Wen et al. (2025). Unsupervised Elicitation of Language Models.
arXiv:2506.10139
```
