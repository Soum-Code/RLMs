import torch
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from collections import deque
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

sys.stdout.reconfigure(encoding='utf-8')

class AdvancedConvergenceDetector:
    def __init__(self, similarity_threshold=0.85, window_size=3):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.similarity_history = deque(maxlen=10)
        self.output_history = deque(maxlen=5)
        # Initialize similarity model once
        self._sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_convergence_metrics(self, outputs_list):
        """Calculate multiple convergence metrics"""
        if len(outputs_list) < 2:
            return {}

        metrics = {
            'semantic_similarities': [],
            'length_variations': [],
            'keyword_stability': [],
            'oscillation_score': 0
        }

        # Calculate pairwise similarities
        for i in range(1, len(outputs_list)):
            sim = self._calculate_semantic_similarity(
                outputs_list[i-1]['output'],
                outputs_list[i]['output']
            )
            metrics['semantic_similarities'].append(sim)

            # Track length variations
            len_diff = abs(len(outputs_list[i]['output']) - len(outputs_list[i-1]['output']))
            metrics['length_variations'].append(len_diff)

        # Calculate keyword stability
        metrics['keyword_stability'] = self._calculate_keyword_stability(outputs_list)

        # Detect oscillation patterns
        metrics['oscillation_score'] = self._detect_oscillation(metrics['semantic_similarities'])

        return metrics

    def _calculate_semantic_similarity(self, text1, text2):
        """Enhanced similarity calculation"""
        if not text1.strip() or not text2.strip():
            return 0.0

        embeddings = self._sim_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def _calculate_keyword_stability(self, outputs_list):
        """Track stability of key concepts across iterations"""
        if len(outputs_list) < 2:
            return []

        keywords_per_output = []
        for output_dict in outputs_list:
            text = output_dict['output'].lower()
            # Extract meaningful keywords (filter common words)
            words = [w.strip('.,!?";') for w in text.split() if len(w) > 3]
            # Remove common stop words and keep unique keywords
            common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was',
'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two',
'who', 'boy', 'did', 'man', 'men', 'run', 'too', 'use', 'any', 'big', 'end', 'far', 'got', 'hot', 'let', 'lot',
'put', 'say', 'she', 'try', 'way', 'win', 'yes'}
            filtered_words = [word for word in words if word not in common_words]
            keywords_per_output.append(set(filtered_words[:8]))  # Top 8 keywords

        stability_scores = []
        for i in range(1, len(keywords_per_output)):
            intersection = len(keywords_per_output[i] & keywords_per_output[i-1])
            union = len(keywords_per_output[i] | keywords_per_output[i-1])
            stability = intersection / union if union > 0 else 0
            stability_scores.append(stability)

        return stability_scores

    def _detect_oscillation(self, similarities):
        """Detect oscillation patterns in similarity scores"""
        if len(similarities) < 3:
            return 0.0

        # Look for alternating high-low-high or low-high-low patterns
        diffs = np.diff(similarities)
        if len(diffs) < 2:
            return 0.0

        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

        # Normalize by sequence length
        oscillation_score = sign_changes / (len(similarities) - 2)
        return float(oscillation_score)

    def statistical_convergence_test(self, similarities):
        """Apply statistical tests for convergence detection"""
        if len(similarities) < 3:
            return {'converged': False, 'confidence': 0.0}

        # Stationarity test using variance
        recent_similarities = similarities[-self.window_size:] if len(similarities) >= self.window_size else similarities
        variance = np.var(recent_similarities)
        mean_similarity = np.mean(recent_similarities)

        # Convergence indicators
        low_variance = variance < 0.05  # Low variance indicates stability
        high_mean = mean_similarity > self.similarity_threshold  # High similarity

        converged = low_variance and high_mean
        confidence = min(1.0, (mean_similarity / self.similarity_threshold) * (1 - variance * 10))

        return {
            'converged': converged,
            'confidence': confidence,
            'variance': variance,
            'mean_similarity': mean_similarity
        }

    def adaptive_stopping_criterion(self, current_iteration, outputs_list, max_iterations=10):
        """Intelligent stopping decision"""
        if current_iteration < 2:
            return {'should_stop': False, 'reason': 'Minimum iterations'}

        if current_iteration >= max_iterations:
            return {'should_stop': True, 'reason': 'Max iterations reached'}

        # Calculate metrics
        metrics = self.calculate_convergence_metrics(outputs_list)

        if not metrics.get('semantic_similarities'):
            return {'should_stop': False, 'reason': 'Insufficient data'}

        # Statistical convergence test
        stat_result = self.statistical_convergence_test(metrics['semantic_similarities'])

        if stat_result['converged'] and stat_result['confidence'] > 0.8:
            return {'should_stop': True, 'reason': 'Statistically converged'}

        # Check for oscillation
        if metrics.get('oscillation_score', 0) > 0.5:
            return {'should_stop': True, 'reason': 'Detected oscillation'}

        # Check for degradation (similarity dropping significantly)
        recent_sims = metrics['semantic_similarities'][-2:]
        if len(recent_sims) >= 2 and recent_sims[-1] < recent_sims[-2] * 0.8:
            return {'should_stop': True, 'reason': 'Quality degradation detected'}

        return {'should_stop': False, 'reason': 'Continue iterating'}

# Enhanced version of your existing RLM class
class ConvergenceRLM:
    def __init__(self, model_name="gpt2"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Handle padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print("Model loaded successfully!")

    def generate_response(self, prompt, max_length=150, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Return only the generated part (remove prompt)
        generated_text = response[len(prompt):].strip()
        return generated_text if generated_text else "[No response generated]"

# Smart experiment runner
class SmartConvergenceExperiment:
    def __init__(self, model_name="gpt2"):
        self.rlm = ConvergenceRLM(model_name)
        self.detector = AdvancedConvergenceDetector()

    def run_smart_experiment(self, initial_prompt, max_iterations=8):
        """Run experiment with intelligent convergence detection"""
        print(f"\n{'='*60}")
        print(f"SMART CONVERGENCE EXPERIMENT")
        print(f"Prompt: {initial_prompt}")
        print(f"{'='*60}")

        outputs = []
        start_time = time.time()

        # First iteration
        current_output = self.rlm.generate_response(initial_prompt)
        outputs.append({
            'iteration': 1,
            'prompt': initial_prompt,
            'output': current_output,
            'timestamp': time.time()
        })

        print(f"\nIteration 1:")
        print(f"Output: {current_output[:150]}...")

        # Subsequent iterations with smart stopping
        for i in range(1, max_iterations):
            reflective_prompt = f"""
Question: {initial_prompt}
Previous answer: {current_output}

Improve this answer by making it more accurate and comprehensive:"""

            new_output = self.rlm.generate_response(reflective_prompt)
            outputs.append({
                'iteration': i+1,
                'prompt': reflective_prompt,
                'output': new_output,
                'timestamp': time.time()
            })

            print(f"\nIteration {i+1}:")
            print(f"Output: {new_output[:150]}...")

            # Apply smart stopping criterion
            stop_decision = self.detector.adaptive_stopping_criterion(i+1, outputs, max_iterations)
            print(f"Stopping decision: {stop_decision['reason']}")

            if stop_decision['should_stop']:
                print(f"🛑 STOPPING at iteration {i+1}")
                break

            current_output = new_output

        # Final analysis
        execution_time = time.time() - start_time
        final_metrics = self.detector.calculate_convergence_metrics(outputs)
        stat_test = self.detector.statistical_convergence_test(
            final_metrics.get('semantic_similarities', [])
        )

        print(f"\n{'='*40}")
        print(f"FINAL ANALYSIS")
        print(f"{'='*40}")
        print(f"Total iterations: {len(outputs)}")
        print(f"Execution time: {execution_time:.2f}s")
        if stat_test:
            print(f"Statistical convergence: Converged={stat_test.get('converged', False)}, Confidence={stat_test.get('confidence', 0.0):.3f}")

        return {
            'outputs': outputs,
            'metrics': final_metrics,
            'statistics': stat_test,
            'execution_time': execution_time
        }

# Utility functions
class ConvergenceUtils:
    @staticmethod
    def quick_convergence_check(outputs_list, threshold=0.85):
        """Quick check if sequence has converged"""
        if len(outputs_list) < 2:
            return False, 0.0

        detector = AdvancedConvergenceDetector()
        similarities = []

        for i in range(1, len(outputs_list)):
            sim = detector._calculate_semantic_similarity(
                outputs_list[i-1]['output'],
                outputs_list[i]['output']
            )
            similarities.append(sim)

        if not similarities:
            return False, 0.0

        avg_similarity = np.mean(similarities)
        return avg_similarity > threshold, avg_similarity

    @staticmethod
    def convergence_report(outputs_list):
        """Generate detailed convergence report"""
        detector = AdvancedConvergenceDetector()
        metrics = detector.calculate_convergence_metrics(outputs_list)

        avg_semantic = np.mean(metrics.get('semantic_similarities', [0])) if metrics.get('semantic_similarities') else 0
        avg_keyword = np.mean(metrics.get('keyword_stability', [0])) if metrics.get('keyword_stability') else 0

        report = {
            'total_iterations': len(outputs_list),
            'average_semantic_similarity': avg_semantic,
            'oscillation_tendency': metrics.get('oscillation_score', 0),
            'keyword_stability': avg_keyword,
            'converged_quick_check': ConvergenceUtils.quick_convergence_check(outputs_list)[0]
        }

        return report

# Main test function
def run_phase2_experiments():
    """Run Phase 2 experiments"""
    experiment_runner = SmartConvergenceExperiment()

    test_prompts = [
        "Explain quantum computing in simple terms",
        "What caused the Great Depression?",
        "Describe the process of photosynthesis"
    ]

    results = {}

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'#'*80}")
        print(f"PHASE 2 EXPERIMENT {i+1}")
        print(f"{'#'*80}")

        try:
            result = experiment_runner.run_smart_experiment(prompt, max_iterations=6)
            results[f'test_{i+1}'] = result

            # Generate and display report
            print(f"\nDETAILED REPORT for '{prompt[:50]}...':")
            report = ConvergenceUtils.convergence_report(result['outputs'])
            for key, value in report.items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"Error in experiment {i+1}: {str(e)}")
            continue

    return results

if __name__ == "__main__":
    print("Starting Phase 2: Convergence Detection Implementation...")
    results = run_phase2_experiments()
    print("\nPhase 2 experiments completed!")
