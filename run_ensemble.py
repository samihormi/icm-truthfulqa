"""Run ensemble ICM to measure uncertainty via disagreement across runs."""
import os
import json
import argparse
from run_icm import load_data, evaluate_with_icm_labels, plot_four_bars
from icm_algorithm import ICMAlgorithm


def run_ensemble_icm(
    api_key, base_url, model, train_data, test_data, 
    n_ensemble=5, max_icm_iters=200, k=6, output_dir='ensemble_runs'
):
    """Run ICM multiple times with different seeds."""
    os.makedirs(output_dir, exist_ok=True)
    
    ensemble_results = []
    
    for seed in range(42, 42 + n_ensemble):
        print(f"\n{'='*70}")
        print(f"ENSEMBLE RUN {seed - 41}/{n_ensemble} (seed={seed})")
        print(f"{'='*70}\n")
        
        # Run ICM
        icm = ICMAlgorithm(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_iterations=max_icm_iters,
            seed=seed
        )
        
        icm_train_labels = icm.run(train_data)
        
        # Evaluate on test set
        icm_accuracy, icm_predictions = evaluate_with_icm_labels(
            icm_train_labels, train_data, test_data,
            api_key, model, base_url, k=k
        )
        
        # Save this run's results
        run_result = {
            'seed': seed,
            'accuracy': icm_accuracy,
            'predictions': icm_predictions,
            'train_labels': {str(k): v for k, v in icm_train_labels.items()}
        }
        
        ensemble_results.append(run_result)
        
        # Save individual run
        run_path = os.path.join(output_dir, f'run_seed{seed}.json')
        with open(run_path, 'w') as f:
            json.dump(run_result, f, indent=2)
        
        print(f"✅ Run {seed - 41}/{n_ensemble} complete: {icm_accuracy*100:.1f}% accuracy")
    
    return ensemble_results


def analyze_ensemble(ensemble_results, output_dir='ensemble_runs'):
    """Analyze disagreement across ensemble and correlation with correctness."""
    import numpy as np
    
    n_test = len(ensemble_results[0]['predictions'])
    n_ensemble = len(ensemble_results)
    
    # For each test example, collect predictions from all runs
    example_analysis = []
    
    for i in range(n_test):
        predictions_for_example = []
        true_label = None
        question_data = None
        
        for run in ensemble_results:
            pred = run['predictions'][i]
            predictions_for_example.append(1 if pred['pred_label'] else 0)
            true_label = pred['true_label']
            question_data = pred
        
        # Calculate ensemble statistics
        predictions_array = np.array(predictions_for_example)
        mean_pred = predictions_array.mean()
        variance = predictions_array.var()
        majority_vote = 1 if mean_pred >= 0.5 else 0
        agreement_count = np.sum(predictions_array == majority_vote)
        
        is_correct = (majority_vote == true_label)
        
        example_analysis.append({
            'question': question_data['question'],
            'choice': question_data['choice'],
            'true_label': true_label,
            'predictions': predictions_for_example,
            'mean_prediction': float(mean_pred),
            'variance': float(variance),
            'disagreement_count': int(n_ensemble - agreement_count),
            'majority_vote': majority_vote,
            'is_correct': is_correct
        })
    
    # Calculate correlation
    variances = [ex['variance'] for ex in example_analysis]
    correctness = [1 if ex['is_correct'] else 0 for ex in example_analysis]
    
    # Group by agreement level
    perfect_agreement = [ex for ex in example_analysis if ex['disagreement_count'] == 0]
    some_disagreement = [ex for ex in example_analysis if 0 < ex['disagreement_count'] < n_ensemble]
    
    print(f"\n{'='*70}")
    print("ENSEMBLE ANALYSIS")
    print(f"{'='*70}")
    print(f"Total test examples: {n_test}")
    print(f"Ensemble size: {n_ensemble}")
    print(f"\nAccuracy by agreement level:")
    print(f"  Perfect agreement (5/5 or 0/5): {len(perfect_agreement)} examples")
    if perfect_agreement:
        acc = sum(ex['is_correct'] for ex in perfect_agreement) / len(perfect_agreement)
        print(f"    Accuracy: {acc*100:.1f}%")
    
    print(f"  Some disagreement (3/2 or 4/1):  {len(some_disagreement)} examples")
    if some_disagreement:
        acc = sum(ex['is_correct'] for ex in some_disagreement) / len(some_disagreement)
        print(f"    Accuracy: {acc*100:.1f}%")
    
    # Save analysis
    analysis_path = os.path.join(output_dir, 'ensemble_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump({
            'summary': {
                'n_test': n_test,
                'n_ensemble': n_ensemble,
                'perfect_agreement_count': len(perfect_agreement),
                'perfect_agreement_accuracy': sum(ex['is_correct'] for ex in perfect_agreement) / len(perfect_agreement) if perfect_agreement else 0,
                'some_disagreement_count': len(some_disagreement),
                'some_disagreement_accuracy': sum(ex['is_correct'] for ex in some_disagreement) / len(some_disagreement) if some_disagreement else 0
            },
            'examples': example_analysis
        }, f, indent=2)
    
    print(f"\n✅ Saved analysis to {analysis_path}")
    print(f"{'='*70}\n")
    
    return example_analysis


def main():
    parser = argparse.ArgumentParser(description='Run Ensemble ICM for uncertainty analysis')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--base_url', type=str, default='https://api.hyperbolic.xyz/v1')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-405B')
    parser.add_argument('--train_data', type=str, default='data/truthfulqa_train.json')
    parser.add_argument('--test_data', type=str, default='data/truthfulqa_test.json')
    parser.add_argument('--output_dir', type=str, default='ensemble_runs')
    parser.add_argument('--n_ensemble', type=int, default=5)
    parser.add_argument('--max_icm_iters', type=int, default=200)
    parser.add_argument('--train_samples', type=int, default=50)
    parser.add_argument('--test_samples', type=int, default=30)
    parser.add_argument('--k', type=int, default=6)
    args = parser.parse_args()
    
    if args.api_key is None:
        args.api_key = os.environ.get('HYPERBOLIC_API_KEY')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_data = load_data(args.train_data)
    if args.train_samples:
        train_data = train_data[:args.train_samples]
    
    test_data = load_data(args.test_data)
    if args.test_samples:
        test_data = test_data[:args.test_samples]
    
    print(f"{'='*70}")
    print("ENSEMBLE ICM - UNCERTAINTY VALIDATION")
    print(f"{'='*70}")
    print(f"Ensemble size: {args.n_ensemble}")
    print(f"Train: {len(train_data)} examples")
    print(f"Test: {len(test_data)} examples")
    print(f"ICM iterations per run: {args.max_icm_iters}")
    print(f"{'='*70}\n")
    
    # Run ensemble
    ensemble_results = run_ensemble_icm(
        args.api_key, args.base_url, args.model,
        train_data, test_data,
        n_ensemble=args.n_ensemble,
        max_icm_iters=args.max_icm_iters,
        k=args.k,
        output_dir=args.output_dir
    )
    
    # Analyze
    analyze_ensemble(ensemble_results, output_dir=args.output_dir)
    
    # Print overall ensemble accuracy
    accuracies = [r['accuracy'] for r in ensemble_results]
    mean_acc = sum(accuracies) / len(accuracies)
    print(f"\n{'='*70}")
    print(f"ENSEMBLE SUMMARY")
    print(f"{'='*70}")
    print(f"Individual run accuracies: {[f'{a*100:.1f}%' for a in accuracies]}")
    print(f"Mean accuracy: {mean_acc*100:.1f}%")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

