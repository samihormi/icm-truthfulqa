# ICM Algorithm 1 - TruthfulQA Implementation

Implementation of Algorithm 1 (Internal Coherence Maximization) from ["Unsupervised Elicitation of Language Models"](https://arxiv.org/abs/2506.10139).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export HYPERBOLIC_API_KEY="your-key-here"

# Run ICM (full dataset)
python run_icm.py --model meta-llama/Meta-Llama-3.1-405B

# Quick test (30 train, 20 test samples)
python run_icm.py --model meta-llama/Meta-Llama-3.1-405B \
  --train_samples 30 --test_samples 20 --max_icm_iters 100
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

## Output

**Files:**
- `outputs/four_bars.png` - Main figure (4 bars on 0-100% scale)
- `outputs/final_results.json` - All metrics
- `outputs/icm_train_labels.json` - ICM-generated labels
- `outputs/icm_predictions.json` - Test predictions

**Graph shows:**
- Random baseline
- Zero-shot baseline
- ICM (ours)
- Golden labels (100%)

## Parameters

```
--model           Base model (default: Meta-Llama-3.1-405B)
--train_samples   Limit train examples (optional)
--test_samples    Limit test examples (optional)
--max_icm_iters   ICM iterations (default: 500)
--k               In-context examples (default: 6)
--seed            Random seed (default: 42)
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
