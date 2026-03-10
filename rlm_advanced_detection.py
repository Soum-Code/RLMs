import numpy as np
import time
import requests
import json
import sys
from scipy import stats
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

# The Ollama API Handler
class OllamaRLM:
    def __init__(self, model_name="gpt-oss:20b", host="http://localhost:11434"):
        print(f"Connecting to Ollama model: {model_name} at {host}")
        self.model_name = model_name
        self.api_url = f"{host}/api/generate"

    def generate_response(self, prompt, max_length=150, temperature=0.7):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_length,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return f"[Error generating response: {e}]"


class AdvancedConvergenceDetector:
    def __init__(self, similarity_threshold=0.85, window_size=3):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.similarity_history = deque(maxlen=10)
        self.output_history = deque(maxlen=5)       

    def calculate_convergence_metrics(self, outputs_list):
        if len(outputs_list) < 2:
            return {}

        metrics = {
            'semantic_similarities': [],
            'length_variations': [],
            'keyword_stability': [],
            'oscillation_score': 0
        }

        for i in range(1, len(outputs_list)):
            sim = self._calculate_semantic_similarity(
                outputs_list[i-1]['output'],
                outputs_list[i]['output']
            )
            metrics['semantic_similarities'].append(sim)

            len_diff = abs(len(outputs_list[i]['output']) - len(outputs_list[i-1]['output']))
            metrics['length_variations'].append(len_diff)

        metrics['keyword_stability'] = self._calculate_keyword_stability(outputs_list)
        metrics['oscillation_score'] = self._detect_oscillation(metrics['semantic_similarities'])

        return metrics

    def _calculate_semantic_similarity(self, text1, text2):
        if not hasattr(self, '_sim_model'):
            self._sim_model = SentenceTransformer('all-MiniLM-L6-v2')

        if not text1.strip() or not text2.strip():
            return 0.0

        embeddings = self._sim_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def _calculate_keyword_stability(self, outputs_list):
        if len(outputs_list) < 2:
            return []

        keywords_per_output = []
        for output_dict in outputs_list:
            text = output_dict['output'].lower()
            words = text.split()
            keywords = [word for word in words if len(word) > 4]
            keywords_per_output.append(set(keywords[:10]))

        stability_scores = []
        for i in range(1, len(keywords_per_output)):
            intersection = len(keywords_per_output[i] & keywords_per_output[i-1])
            union = len(keywords_per_output[i] | keywords_per_output[i-1])
            stability = intersection / union if union > 0 else 0
            stability_scores.append(stability)

        return stability_scores

    def _detect_oscillation(self, similarities):
        if len(similarities) < 3:
            return 0.0

        diffs = np.diff(similarities)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

        oscillation_score = sign_changes / (len(similarities) - 2)
        return float(oscillation_score)

    def statistical_convergence_test(self, similarities):
        if len(similarities) < 3:
            return {'converged': False, 'confidence': 0.0, 'variance': 0.0, 'mean_similarity': 0.0}

        recent_similarities = similarities[-self.window_size:]
        variance = np.var(recent_similarities)
        mean_similarity = np.mean(recent_similarities)

        low_variance = variance < 0.05
        high_mean = mean_similarity > self.similarity_threshold

        converged = low_variance and high_mean
        confidence = min(1.0, (mean_similarity / self.similarity_threshold) * (1 - variance * 10))

        return {
            'converged': converged,
            'confidence': confidence,
            'variance': variance,
            'mean_similarity': mean_similarity
        }

    def adaptive_stopping_criterion(self, current_iteration, outputs_list, max_iterations=10):
        if current_iteration < 2:
            return {'should_stop': False, 'reason': 'Minimum iterations'}

        if current_iteration >= max_iterations:
            return {'should_stop': True, 'reason': 'Max iterations reached'}

        recent_outputs = outputs_list[-3:] if len(outputs_list) >= 3 else outputs_list
        metrics = self.calculate_convergence_metrics(outputs_list)

        if not metrics.get('semantic_similarities'):
            return {'should_stop': False, 'reason': 'Insufficient data'}

        stat_result = self.statistical_convergence_test(metrics['semantic_similarities'])

        if stat_result['converged'] and stat_result['confidence'] > 0.8:
            return {'should_stop': True, 'reason': 'Statistically converged'}

        if metrics.get('oscillation_score', 0) > 0.5:
            return {'should_stop': True, 'reason': 'Detected oscillation'}

        recent_sims = metrics['semantic_similarities'][-2:]
        if len(recent_sims) >= 2 and recent_sims[-1] < recent_sims[-2] * 0.8:
            return {'should_stop': True, 'reason': 'Quality degradation detected'}

        return {'should_stop': False, 'reason': 'Continue iterating'}


class SmartConvergenceExperiment:
    def __init__(self, model_name="gpt-oss:20b"):
        self.rlm = OllamaRLM(model_name)
        self.detector = AdvancedConvergenceDetector()

    def run_smart_experiment(self, initial_prompt, max_iterations=8):
        print(f"\n{'='*60}")
        print(f"SMART CONVERGENCE EXPERIMENT")
        print(f"Prompt: {initial_prompt}")
        print(f"{'='*60}")

        outputs = []
        all_metrics = []

        start_time = time.time()
        current_output = self.rlm.generate_response(initial_prompt)
        outputs.append({
            'iteration': 1,
            'prompt': initial_prompt,
            'output': current_output,
            'timestamp': time.time()
        })

        print(f"\nIteration 1:")
        print(f"Output: {current_output[:200]}...")

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
            print(f"Output: {new_output[:200]}...")

            stop_decision = self.detector.adaptive_stopping_criterion(i+1, outputs, max_iterations)
            print(f"Stopping decision: {stop_decision['reason']}")

            if stop_decision['should_stop']:
                print(f"🛑 STOPPING at iteration {i+1}")
                break

            current_output = new_output

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
        print(f"Statistical convergence: {stat_test}")

        return {
            'outputs': outputs,
            'metrics': final_metrics,
            'statistics': stat_test,
            'execution_time': execution_time
        }


class ConvergenceUtils:
    @staticmethod
    def quick_convergence_check(outputs_list, threshold=0.85):
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
        detector = AdvancedConvergenceDetector()
        metrics = detector.calculate_convergence_metrics(outputs_list)

        report = {
            'total_iterations': len(outputs_list),
            'average_semantic_similarity': np.mean(metrics.get('semantic_similarities', [0])) if metrics.get('semantic_similarities') else 0,
            'oscillation_tendency': metrics.get('oscillation_score', 0),
            'keyword_stability': np.mean(metrics.get('keyword_stability', [0])) if metrics.get('keyword_stability') else 0,
            'converged_quick_check': ConvergenceUtils.quick_convergence_check(outputs_list)[0]
        }

        return report

def compare_detection_methods():
    experiment_runner = SmartConvergenceExperiment()

    test_prompts = [
        "Explain quantum computing in simple terms",
        "What caused the Great Depression?",
        "Describe the process of photosynthesis"
    ]

    comparison_results = {}

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'#'*80}")
        print(f"COMPARISON TEST {i+1}")
        print(f"{'#'*80}")

        result = experiment_runner.run_smart_experiment(prompt, max_iterations=6)
        comparison_results[f'test_{i+1}'] = result

        print(f"\nResults saved for '{prompt[:50]}...'")

    return comparison_results


if __name__ == "__main__":
    print("Running smart convergence detection experiments against Ollama (gpt-oss:20b)...")
    results = compare_detection_methods()

    for test_name, result in results.items():
        print(f"\n{test_name.upper()} REPORT:")
        report = ConvergenceUtils.convergence_report(result['outputs'])
        for key, value in report.items():
            print(f"  {key}: {value}")
