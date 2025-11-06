"""
Complete implementation of Algorithm 1 (ICM) from the paper.

This implements Internal Coherence Maximization without consistency fix.
"""
import json
import random
import math
import numpy as np
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from tqdm import tqdm


class ICMAlgorithm:
    """Internal Coherence Maximization (Algorithm 1 from paper)."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.hyperbolic.xyz/v1",
        model: str = "meta-llama/Meta-Llama-3.1-405B",
        alpha: float = 50.0,
        T_0: float = 10.0,
        T_min: float = 0.01,
        beta: float = 0.99,
        K_init: int = 20,
        max_iterations: int = 1000,
        seed: int = 42
    ):
        """
        Initialize ICM algorithm.
        
        Args:
            api_key: Hyperbolic API key
            base_url: API base URL
            model: BASE model name (not instruct)
            alpha: Weight for mutual predictability (default 50 from paper)
            T_0: Initial temperature (default 10 from paper)
            T_min: Final temperature (default 0.01 from paper)
            beta: Cooling rate (default 0.99 from paper)
            K_init: Number of initial random labels (default 20 from paper)
            max_iterations: Maximum iterations (default 1000)
            seed: Random seed
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.alpha = alpha
        self.T_0 = T_0
        self.T_min = T_min
        self.beta = beta
        self.K_init = K_init
        self.max_iterations = max_iterations
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"Initialized ICM with:")
        print(f"  Model: {model}")
        print(f"  α={alpha}, T_0={T_0}, T_min={T_min}, β={beta}")
        print(f"  K_init={K_init}, max_iter={max_iterations}")
    
    def format_example(self, item: Dict[str, Any], label: bool) -> str:
        """Format a single example for the prompt."""
        label_text = "True" if label else "False"
        return f"""Question: {item['question']}
Claim: {item['choice']}
I think this Claim is {label_text}"""
    
    def build_prompt(self, labeled_data: Dict[int, Dict], target_item: Dict[str, Any]) -> str:
        """
        Build in-context prompt with all labeled examples except target.
        
        Args:
            labeled_data: Dict of {index: {"item": ..., "label": ...}}
            target_item: The item we're predicting for
            
        Returns:
            Prompt string ending with "I think this Claim is"
        """
        # Include all labeled examples as context
        context_examples = []
        for idx, data in labeled_data.items():
            context_examples.append(self.format_example(data["item"], data["label"]))
        
        # Add target question without label
        target_text = f"""Question: {target_item['question']}
Claim: {target_item['choice']}
I think this Claim is"""
        
        # Combine with newlines
        if context_examples:
            prompt = "\n\n".join(context_examples) + "\n\n" + target_text
        else:
            prompt = target_text
        
        return prompt
    
    def get_label_probability(self, prompt: str) -> Tuple[float, float]:
        """
        Get P(True) and P(False) for next token given prompt.
        
        Returns:
            (P_true, P_false) normalized probabilities
        """
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=1,
                temperature=0,
                logprobs=10,  # Get top 10 to catch case variations
                echo=False
            )
            
            # Extract logprobs
            top_logprobs = response.choices[0].logprobs.top_logprobs[0]
            
            # Find True and False tokens (check case variations)
            log_p_true = float('-inf')
            log_p_false = float('-inf')
            
            for token, logprob in top_logprobs.items():
                token_lower = token.strip().lower()
                if token_lower == "true":
                    log_p_true = max(log_p_true, logprob)
                elif token_lower == "false":
                    log_p_false = max(log_p_false, logprob)
            
            # Convert to probabilities and normalize
            p_true = math.exp(log_p_true)
            p_false = math.exp(log_p_false)
            total = p_true + p_false
            
            if total > 0:
                p_true /= total
                p_false /= total
            else:
                # Fallback: equal probabilities if neither found
                p_true = 0.5
                p_false = 0.5
            
            return p_true, p_false
            
        except Exception as e:
            print(f"Error getting label probability: {e}")
            return 0.5, 0.5
    
    def calculate_conditional_log_prob(
        self, 
        labeled_data: Dict[int, Dict],
        target_idx: int,
        target_item: Dict[str, Any],
        target_label: bool
    ) -> float:
        """
        Calculate log P(y_i | x_i, D_{-i}).
        
        Args:
            labeled_data: All labeled examples except target
            target_idx: Index of target (for excluding from context)
            target_item: The target item
            target_label: The label to evaluate
            
        Returns:
            Log probability of target_label
        """
        # Build context with all examples except target
        context = {k: v for k, v in labeled_data.items() if k != target_idx}
        prompt = self.build_prompt(context, target_item)
        
        # Get probabilities
        p_true, p_false = self.get_label_probability(prompt)
        
        # Return log probability of target label
        p = p_true if target_label else p_false
        return math.log(max(p, 1e-10))  # Avoid log(0)
    
    def calculate_score(self, labeled_data: Dict[int, Dict], unlabeled_items: List[Dict]) -> float:
        """
        Calculate U(D) = α * P_θ(D).
        
        Note: No inconsistency penalty per instructions.
        
        Args:
            labeled_data: Dict of {index: {"item": ..., "label": ...}}
            unlabeled_items: Full list of items (for index reference)
            
        Returns:
            Score U(D)
        """
        if len(labeled_data) == 0:
            return 0.0
        
        # Calculate mutual predictability: average log prob
        total_log_prob = 0.0
        count = 0
        
        for idx, data in labeled_data.items():
            log_prob = self.calculate_conditional_log_prob(
                labeled_data, idx, data["item"], data["label"]
            )
            total_log_prob += log_prob
            count += 1
        
        # P_θ(D) = average log probability
        mutual_predictability = total_log_prob / count if count > 0 else 0.0
        
        # U(D) = α * P_θ(D)  (no inconsistency penalty)
        score = self.alpha * mutual_predictability
        
        return score
    
    def run(self, unlabeled_items: List[Dict[str, Any]]) -> Dict[int, bool]:
        """
        Run Algorithm 1 (ICM) to generate labels.
        
        Args:
            unlabeled_items: List of {"question": ..., "choice": ..., ...}
            
        Returns:
            Dict of {index: label (bool)} for all items
        """
        print(f"\n=== Running ICM on {len(unlabeled_items)} examples ===\n")
        
        # Step 1: Initialize with K random labels
        print(f"Step 1: Initializing with {self.K_init} random labels...")
        labeled_data = {}
        indices = list(range(len(unlabeled_items)))
        random.shuffle(indices)
        
        for i in range(min(self.K_init, len(unlabeled_items))):
            idx = indices[i]
            labeled_data[idx] = {
                "item": unlabeled_items[idx],
                "label": random.choice([True, False])
            }
        
        # Calculate initial score
        current_score = self.calculate_score(labeled_data, unlabeled_items)
        print(f"Initial score: {current_score:.4f} with {len(labeled_data)} labeled")
        
        # Step 2: Main search loop
        print(f"\nStep 2: Running simulated annealing for {self.max_iterations} iterations...\n")
        
        best_score = current_score
        best_labeled_data = labeled_data.copy()
        accepts = 0
        rejects = 0
        
        for n in tqdm(range(1, self.max_iterations + 1), desc="ICM Search"):
            # Update temperature
            T = max(self.T_min, self.T_0 / (1 + self.beta * math.log(n)))
            
            # Sample an example to label
            # Prioritize unlabeled, but can re-label existing
            unlabeled_indices = [i for i in range(len(unlabeled_items)) if i not in labeled_data]
            if unlabeled_indices and random.random() < 0.8:  # 80% chance to pick unlabeled
                target_idx = random.choice(unlabeled_indices)
            else:
                target_idx = random.choice(list(range(len(unlabeled_items))))
            
            target_item = unlabeled_items[target_idx]
            
            # Assign label: ŷ_i = arg max P_θ(y_i | x_i, D \ {(x_i, y_i)})
            context = {k: v for k, v in labeled_data.items() if k != target_idx}
            prompt = self.build_prompt(context, target_item)
            p_true, p_false = self.get_label_probability(prompt)
            
            # Choose label with higher probability
            new_label = (p_true > p_false)
            
            # Create new labeled data
            new_labeled_data = labeled_data.copy()
            new_labeled_data[target_idx] = {
                "item": target_item,
                "label": new_label
            }
            
            # Calculate new score
            new_score = self.calculate_score(new_labeled_data, unlabeled_items)
            
            # Accept/reject based on simulated annealing
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                # Accept
                labeled_data = new_labeled_data
                current_score = new_score
                accepts += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_labeled_data = labeled_data.copy()
            else:
                # Reject
                rejects += 1
            
            # Logging
            if n % 100 == 0:
                accept_rate = accepts / (accepts + rejects) if (accepts + rejects) > 0 else 0
                print(f"  Iter {n}: score={current_score:.4f}, best={best_score:.4f}, "
                      f"T={T:.4f}, labeled={len(labeled_data)}/{len(unlabeled_items)}, "
                      f"accept_rate={accept_rate:.2f}")
                accepts, rejects = 0, 0
            
            # Early stopping if all labeled
            if len(labeled_data) >= len(unlabeled_items):
                print(f"\n  All examples labeled at iteration {n}. Stopping.")
                break
        
        print(f"\n=== ICM Complete ===")
        print(f"Final score: {best_score:.4f}")
        print(f"Labeled: {len(best_labeled_data)}/{len(unlabeled_items)}")
        
        # Return labels only
        labels = {idx: data["label"] for idx, data in best_labeled_data.items()}
        return labels

