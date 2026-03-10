import torch
import numpy as np
from typing import List, Dict
import re
import random
import time
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding='utf-8')

# First, let's make sure we have the required classes from previous phases
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
        generated_text = response[len(prompt):].strip()
        return generated_text if generated_text else "[No response generated]"

class AdvancedConvergenceDetector:
    def __init__(self, similarity_threshold=0.85, window_size=3):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self._sim_model = SentenceTransformer('all-MiniLM-L6-v2')

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
            words = [w.strip('.,!?";') for w in text.split() if len(w) > 3]
            common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was',
'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two',
'who', 'boy', 'did', 'man', 'men', 'run', 'too', 'use', 'any', 'big', 'end', 'far', 'got', 'hot', 'let', 'lot',
'put', 'say', 'she', 'try', 'way', 'win', 'yes'}
            filtered_words = [word for word in words if word not in common_words]
            keywords_per_output.append(set(filtered_words[:8]))

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
        if len(diffs) < 2:
            return 0.0

        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        oscillation_score = sign_changes / (len(similarities) - 2)
        return float(oscillation_score)

# Now the main Phase 3 implementation
class ConvergenceIntervention:
    def __init__(self):
        self.intervention_history = []
        self.successful_interventions = {}

    def adaptive_depth_control(self, current_iteration: int, outputs_list: List[Dict],
                             max_allowed_depth: int = 10) -> Dict:
        """Dynamically adjust recursion depth based on convergence signals"""

        if len(outputs_list) < 3:
            return {'continue_recursion': True, 'reason': 'Insufficient data for assessment'}

        # Get recent convergence metrics
        detector = AdvancedConvergenceDetector()
        metrics = detector.calculate_convergence_metrics(outputs_list)

        # Check if we're plateauing
        recent_sims = metrics.get('semantic_similarities', [])
        if len(recent_sims) >= 3:
            trend = np.polyfit(range(len(recent_sims)), recent_sims, 1)[0]

            # If improvement is slowing down significantly
            if trend < 0.01 and max(recent_sims) > 0.7:
                return {'continue_recursion': False, 'reason': 'Diminishing returns detected'}

        # Check for oscillation
        if metrics.get('oscillation_score', 0) > 0.4:
            return {'continue_recursion': False, 'reason': 'Oscillation risk detected'}

        # Check maximum depth
        if current_iteration >= max_allowed_depth:
            return {'continue_recursion': False, 'reason': 'Maximum depth reached'}

        return {'continue_recursion': True, 'reason': 'Continue recursion'}

    def dynamic_prompt_modification(self, original_prompt: str, current_output: str,
                                  iteration: int, convergence_metrics: Dict) -> str:
        """Modify prompts based on convergence state"""

        base_instruction = f"""Original task: {original_prompt}

Current answer attempt #{iteration}: {current_output}

Instructions for improvement:"""

        # Analyze convergence state and adapt instruction
        recent_sims = convergence_metrics.get('semantic_similarities', [])

        if len(recent_sims) >= 2:
            improvement_rate = recent_sims[-1] - recent_sims[-2]

            if improvement_rate > 0.1:
                instruction = f"{base_instruction} Continue refining with emphasis on depth and detail."
            elif improvement_rate < -0.05:
                instruction = f"{base_instruction} Backtrack and correct errors. Focus on accuracy over complexity."
            elif len(recent_sims) > 3 and np.std(recent_sims[-3:]) < 0.05:
                instruction = f"{base_instruction} Try a completely different approach. Think outside the box."
            else:
                instruction = f"{base_instruction} Improve accuracy and completeness."
        else:
            instruction = f"{base_instruction} Improve accuracy and completeness."

        return instruction

    def confidence_based_intervention(self, outputs_list: List[Dict]) -> Dict:
        """Intervene based on confidence estimation"""
        if len(outputs_list) < 2:
            return {'intervention_needed': False, 'type': 'none'}

        detector = AdvancedConvergenceDetector()
        metrics = detector.calculate_convergence_metrics(outputs_list)

        recent_sims = metrics.get('semantic_similarities', [])
        if len(recent_sims) >= 3:
            confidence_std = np.std(recent_sims[-3:])
            avg_confidence = np.mean(recent_sims[-3:])

            if confidence_std < 0.03 and avg_confidence > 0.8:
                return {'intervention_needed': False, 'type': 'none'}
            elif confidence_std > 0.15:
                return {'intervention_needed': True, 'type': 'reset_and_refocus'}
            elif avg_confidence < 0.5:
                return {'intervention_needed': True, 'type': 'diversity_boost'}

        return {'intervention_needed': False, 'type': 'none'}

    def diversity_intervention(self, outputs_list: List[Dict]) -> str:
        """Inject diversity when stuck in local optima"""
        if not outputs_list:
            return ""

        latest_output = outputs_list[-1]['output']

        diversity_prompts = [
            f"Reframe the following answer from a completely different perspective:\n{latest_output}",
            f"Challenge the assumptions in this answer and provide alternatives:\n{latest_output}",
            f"Consider this problem from the viewpoint of a domain expert who disagrees:\n{latest_output}",
            f"What would someone with opposite views say about this answer?\n{latest_output}"
        ]

        return random.choice(diversity_prompts)

    def error_correction_intervention(self, outputs_list: List[Dict]) -> str:
        """Active error detection and correction"""
        if len(outputs_list) < 2:
            return ""

        current_output = outputs_list[-1]['output']

        error_prompts = [
            f"Review this answer for factual errors and logical inconsistencies:\n{current_output}",
            f"Identify potential mistakes in reasoning:\n{current_output}",
            f"Critically evaluate this response for accuracy:\n{current_output}",
            f"Spot-check facts and figures in this answer:\n{current_output}"
        ]

        return random.choice(error_prompts)

class InterventionAwareRLM:
    def __init__(self, model_name="gpt2"):
        self.rlm = ConvergenceRLM(model_name)
        self.intervention_system = ConvergenceIntervention()
        self.iteration_history = []

    def run_intervention_experiment(self, initial_prompt: str, max_iterations: int = 8) -> Dict:
        """Run experiment with active interventions"""
        print(f"\n{'='*60}")
        print(f"INTERVENTION-AWARE CONVERGENCE EXPERIMENT")
        print(f"Prompt: {initial_prompt}")
        print(f"{'='*60}")

        outputs = []
        interventions_applied = []

        # First iteration
        start_time = time.time()
        current_output = self.rlm.generate_response(initial_prompt)
        outputs.append({
            'iteration': 1,
            'prompt': initial_prompt,
            'output': current_output,
            'intervention': 'none'
        })

        print(f"\nIteration 1:")
        print(f"Output: {current_output[:150]}...")

        # Subsequent iterations with interventions
        for i in range(1, max_iterations):
            # Calculate current convergence metrics
            detector = AdvancedConvergenceDetector()
            metrics = detector.calculate_convergence_metrics(outputs)

            # Check if intervention is needed
            confidence_intervention = self.intervention_system.confidence_based_intervention(outputs)
            intervention_type = 'none'
            effective_prompt = ""

            if confidence_intervention['intervention_needed']:
                intervention_type = confidence_intervention['type']

                if intervention_type == 'diversity_boost':
                    effective_prompt = self.intervention_system.diversity_intervention(outputs)
                    print(f"🔄 Applying diversity intervention...")
                elif intervention_type == 'reset_and_refocus':
                    effective_prompt = self.intervention_system.error_correction_intervention(outputs)
                    print(f"🔧 Applying error correction intervention...")
                else:
                    effective_prompt = self.intervention_system.dynamic_prompt_modification(
                        initial_prompt, current_output, i+1, metrics
                    )
            else:
                # Normal improvement cycle
                effective_prompt = self.intervention_system.dynamic_prompt_modification(
                    initial_prompt, current_output, i+1, metrics
                )

            # Apply adaptive depth control
            depth_decision = self.intervention_system.adaptive_depth_control(i+1, outputs, max_iterations)
            if not depth_decision['continue_recursion']:
                print(f"🛑 {depth_decision['reason']} - Stopping recursion")
                break

            # Generate response
            new_output = self.rlm.generate_response(effective_prompt)
            outputs.append({
                'iteration': i+1,
                'prompt': effective_prompt,
                'output': new_output,
                'intervention': intervention_type
            })

            interventions_applied.append(intervention_type)
            print(f"\nIteration {i+1} (Intervention: {intervention_type}):")
            print(f"Output: {new_output[:150]}...")

            current_output = new_output

        # Final analysis
        execution_time = time.time() - start_time
        final_detector = AdvancedConvergenceDetector()
        final_metrics = final_detector.calculate_convergence_metrics(outputs)

        print(f"\n{'='*40}")
        print(f"FINAL INTERVENTION ANALYSIS")
        print(f"{'='*40}")
        print(f"Total iterations: {len(outputs)}")
        print(f"Interventions applied: {interventions_applied}")
        print(f"Execution time: {execution_time:.2f}s")

        return {
            'outputs': outputs,
            'interventions': interventions_applied,
            'metrics': final_metrics,
            'execution_time': execution_time
        }

# Simple baseline for comparison
class SimpleBaselineRLM:
    def __init__(self, model_name="gpt2"):
        self.rlm = ConvergenceRLM(model_name)

    def run_baseline_experiment(self, initial_prompt: str, max_iterations: int = 8) -> Dict:
        """Run baseline experiment without interventions"""
        print(f"\n{'='*60}")
        print(f"BASELINE CONVERGENCE EXPERIMENT")
        print(f"Prompt: {initial_prompt}")
        print(f"{'='*60}")

        outputs = []
        start_time = time.time()

        # First iteration
        current_output = self.rlm.generate_response(initial_prompt)
        outputs.append({
            'iteration': 1,
            'prompt': initial_prompt,
            'output': current_output
        })

        print(f"\nIteration 1:")
        print(f"Output: {current_output[:150]}...")

        # Subsequent iterations
        for i in range(1, max_iterations):
            reflective_prompt = f"""
Question: {initial_prompt}
Previous answer: {current_output}

Improve this answer by making it more accurate and comprehensive:"""

            new_output = self.rlm.generate_response(reflective_prompt)
            outputs.append({
                'iteration': i+1,
                'prompt': reflective_prompt,
                'output': new_output
            })

            print(f"\nIteration {i+1}:")
            print(f"Output: {new_output[:150]}...")

            current_output = new_output

        execution_time = time.time() - start_time
        detector = AdvancedConvergenceDetector()
        final_metrics = detector.calculate_convergence_metrics(outputs)

        return {
            'outputs': outputs,
            'metrics': final_metrics,
            'execution_time': execution_time
        }

# Main test function
def run_phase3_comparison():
    """Run Phase 3 intervention comparison experiments"""

    test_prompts = [
        "Explain quantum computing in simple terms",
        "What are the main causes of climate change?",
        "Describe the process of evolution by natural selection"
    ]

    results = {}

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'#'*100}")
        print(f"PHASE 3 COMPARISON EXPERIMENT {i+1}")
        print(f"{'#'*100}")

        try:
            # Run baseline
            print("\n🧪 Running Baseline Approach...")
            baseline_rlm = SimpleBaselineRLM()
            baseline_result = baseline_rlm.run_baseline_experiment(prompt, max_iterations=5)

            # Run intervention approach
            print("\n🚀 Running Intervention Approach...")
            intervention_rlm = InterventionAwareRLM()
            intervention_result = intervention_rlm.run_intervention_experiment(prompt, max_iterations=5)

            # Compare results
            baseline_detector = AdvancedConvergenceDetector()
            baseline_report = baseline_detector.calculate_convergence_metrics(baseline_result['outputs'])
            intervention_report = baseline_detector.calculate_convergence_metrics(intervention_result['outputs'])

            avg_baseline_sim = np.mean(baseline_report.get('semantic_similarities', [0])) if baseline_report.get('semantic_similarities') else 0
            avg_intervention_sim = np.mean(intervention_report.get('semantic_similarities', [0])) if intervention_report.get('semantic_similarities') else 0

            comparison = {
                'baseline': {
                    'iterations': len(baseline_result['outputs']),
                    'execution_time': baseline_result['execution_time'],
                    'avg_similarity': avg_baseline_sim
                },
                'intervention': {
                    'iterations': len(intervention_result['outputs']),
                    'execution_time': intervention_result['execution_time'],
                    'avg_similarity': avg_intervention_sim,
                    'interventions_used': intervention_result['interventions']
                },
                'improvement': avg_intervention_sim - avg_baseline_sim
            }

            print(f"\n📊 COMPARISON RESULTS:")
            print(f"Baseline iterations: {comparison['baseline']['iterations']}")
            print(f"Intervention iterations: {comparison['intervention']['iterations']}")
            print(f"Baseline convergence: {comparison['baseline']['avg_similarity']:.3f}")
            print(f"Intervention convergence: {comparison['intervention']['avg_similarity']:.3f}")
            print(f"Improvement: {comparison['improvement']:.3f}")
            print(f"Interventions used: {comparison['intervention']['interventions_used']}")

            results[f'experiment_{i+1}'] = comparison

        except Exception as e:
            print(f"Error in experiment {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return results

if __name__ == "__main__":
    print("Starting Phase 3: Intervention Strategies Implementation...")
    try:
        results = run_phase3_comparison()
        print(f"\nPhase 3 experiments completed! Results: {len(results)} successful experiments")
    except Exception as e:
        print(f"Error running Phase 3: {str(e)}")
        import traceback
        traceback.print_exc()
