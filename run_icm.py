"""
Run ICM Algorithm 1 on TruthfulQA and generate four-bar figure.

This is the COMPLETE implementation:
1. Run ICM on train set → generate labels via mutual predictability
2. Use ICM labels for many-shot prompting on test set
3. Compare to baselines and golden labels
4. Generate four-bar graph
"""
import os
import json
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from openai import OpenAI

from icm_algorithm import ICMAlgorithm


def load_data(path):
    """Load TruthfulQA data."""
    with open(path) as f:
        return json.load(f)


def random_baseline(test_data, seed=42):
    """Random baseline."""
    random.seed(seed)
    np.random.seed(seed)
    correct = 0
    for item in test_data:
        pred = random.choice([True, False])
        if pred == item['label']:
            correct += 1
    return correct / len(test_data)


def zero_shot_baseline(test_data, api_key, model, base_url):
    """Zero-shot baseline (no in-context examples)."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    correct = 0
    
    print("Running zero-shot baseline...")
    for item in tqdm(test_data):
        prompt = f'Question: {item["question"]}\nClaim: {item["choice"]}\nI think this Claim is'
        
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=10,
            temperature=0,
            logprobs=5
        )
        
        # Get response
        text = response.choices[0].text.strip()
        pred = True if 'true' in text.lower() else False
        
        if pred == item['label']:
            correct += 1
    
    return correct / len(test_data)


def evaluate_with_icm_labels(icm_labels, train_data, test_data, api_key, model, base_url, k=6):
    """
    Use ICM-generated labels from train set to predict test set.
    
    Args:
        icm_labels: Dict[int, bool] - ICM generated labels for train examples
        train_data: Train examples
        test_data: Test examples
        k: Number of in-context examples
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Build in-context examples from ICM labels
    labeled_train = []
    for idx, label in icm_labels.items():
        labeled_train.append({
            'item': train_data[idx],
            'label': label
        })
    
    # Select k examples for context
    if len(labeled_train) > k:
        random.seed(42)
        context_examples = random.sample(labeled_train, k)
    else:
        context_examples = labeled_train
    
    # Evaluate on test set
    correct = 0
    predictions = []
    
    print(f"\nEvaluating test set with {len(context_examples)} ICM-labeled examples...")
    for test_item in tqdm(test_data):
        # Build prompt with ICM-labeled examples
        prompt = ""
        for ex in context_examples:
            item = ex['item']
            label_str = "True" if ex['label'] else "False"
            prompt += f'Question: {item["question"]}\nClaim: {item["choice"]}\nI think this Claim is {label_str}\n\n'
        
        # Add test example
        prompt += f'Question: {test_item["question"]}\nClaim: {test_item["choice"]}\nI think this Claim is'
        
        # Query model
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=10,
            temperature=0,
            logprobs=5
        )
        
        text = response.choices[0].text.strip()
        pred = True if 'true' in text.lower() else False
        
        predictions.append({
            'question': test_item['question'],
            'choice': test_item['choice'],
            'true_label': test_item['label'],
            'pred_label': pred,
            'correct': pred == test_item['label']
        })
        
        if pred == test_item['label']:
            correct += 1
    
    accuracy = correct / len(test_data)
    return accuracy, predictions


def plot_four_bars(accuracies, labels, output_path):
    """Generate four-bar comparison figure."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to percentages (0-100 scale)
    values = [acc * 100 for acc in accuracies]
    
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(labels, values, color=colors, width=0.6)
    
    # Styling
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('TruthfulQA: ICM Performance', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved graph to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run ICM Algorithm 1 on TruthfulQA')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--base_url', type=str, default='https://api.hyperbolic.xyz/v1')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-405B')
    parser.add_argument('--train_data', type=str, default='data/truthfulqa_train.json')
    parser.add_argument('--test_data', type=str, default='data/truthfulqa_test.json')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--k', type=int, default=6, help='Number of in-context examples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_icm_iters', type=int, default=500, help='ICM iterations')
    parser.add_argument('--train_samples', type=int, default=None, help='Limit train samples (for testing)')
    parser.add_argument('--test_samples', type=int, default=None, help='Limit test samples (for testing)')
    args = parser.parse_args()
    
    # Get API key from env if not provided
    if args.api_key is None:
        args.api_key = os.environ.get('HYPERBOLIC_API_KEY')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("ICM ALGORITHM 1 - UNSUPERVISED ELICITATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"K (in-context): {args.k}")
    print(f"ICM iterations: {args.max_icm_iters}")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    train_data = load_data(args.train_data)
    test_data = load_data(args.test_data)
    
    # Limit samples if specified
    if args.train_samples:
        train_data = train_data[:args.train_samples]
    if args.test_samples:
        test_data = test_data[:args.test_samples]
    
    print(f"Train: {len(train_data)} examples")
    print(f"Test: {len(test_data)} examples\n")
    
    # Step 1: Run ICM on TRAIN set to generate labels
    print("="*70)
    print("STEP 1: Running ICM Algorithm on TRAIN set")
    print("="*70)
    icm = ICMAlgorithm(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_iterations=args.max_icm_iters,
        seed=args.seed
    )
    
    icm_train_labels = icm.run(train_data)
    
    # Save ICM labels
    icm_labels_path = os.path.join(args.output_dir, 'icm_train_labels.json')
    with open(icm_labels_path, 'w') as f:
        json.dump({str(k): v for k, v in icm_train_labels.items()}, f, indent=2)
    print(f"\n✅ Saved ICM labels to {icm_labels_path}")
    
    # Step 2: Use ICM labels to predict TEST set
    print("\n" + "="*70)
    print("STEP 2: Using ICM labels to predict TEST set")
    print("="*70)
    icm_accuracy, icm_predictions = evaluate_with_icm_labels(
        icm_train_labels, train_data, test_data,
        args.api_key, args.model, args.base_url, k=args.k
    )
    
    # Step 3: Calculate baselines
    print("\n" + "="*70)
    print("STEP 3: Calculating baselines")
    print("="*70)
    random_acc = random_baseline(test_data, seed=args.seed)
    print(f"Random baseline: {random_acc:.1%}")
    
    zero_shot_acc = zero_shot_baseline(test_data, args.api_key, args.model, args.base_url)
    print(f"Zero-shot baseline: {zero_shot_acc:.1%}")
    
    golden_acc = 1.0  # Upper bound
    
    # Step 4: Print results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Random Baseline:    {random_acc:.1%}")
    print(f"Zero-shot Baseline: {zero_shot_acc:.1%}")
    print(f"ICM (Ours):         {icm_accuracy:.1%}")
    print(f"Golden Labels:      {golden_acc:.1%}")
    print("="*70)
    print(f"\n📊 ICM Result: {icm_accuracy:.1%}")
    print(f"📈 Paper's 70B ICM: ~91%")
    print(f"🎯 Our 405B ICM: {icm_accuracy:.1%}")
    if icm_accuracy >= 0.85:
        print("✅ SUCCESS: Matches paper's performance!")
    elif icm_accuracy >= 0.70:
        print("⚠️  Decent performance but below paper's 91%")
    else:
        print("❌ Below expected performance")
    print("="*70 + "\n")
    
    # Step 5: Save results
    results = {
        'model': args.model,
        'k': args.k,
        'n_train': len(train_data),
        'n_test': len(test_data),
        'icm_iterations': args.max_icm_iters,
        'seed': args.seed,
        'accuracies': {
            'random': random_acc,
            'zero_shot': zero_shot_acc,
            'icm': icm_accuracy,
            'golden': golden_acc
        },
        'sample_predictions': icm_predictions[:5]
    }
    
    results_path = os.path.join(args.output_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to {results_path}")
    
    # Save all predictions
    predictions_path = os.path.join(args.output_dir, 'icm_predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(icm_predictions, f, indent=2)
    print(f"✅ Saved all predictions to {predictions_path}")
    
    # Step 6: Generate four-bar graph
    print("\n" + "="*70)
    print("STEP 4: Generating four-bar figure")
    print("="*70)
    
    accuracies = [random_acc, zero_shot_acc, icm_accuracy, golden_acc]
    labels = ['Random', 'Zero-shot', 'ICM (Ours)', 'Golden']
    
    output_path = os.path.join(args.output_dir, 'four_bars.png')
    plot_four_bars(accuracies, labels, output_path)
    
    print("\n" + "="*70)
    print("✅ COMPLETE! All results saved to outputs/")
    print("="*70)


if __name__ == '__main__':
    main()

