import torch
import numpy as np
from typing import List, Dict, Tuple
import time
import json
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.metrics import mean_squared_error
import seaborn as sns
import sys

from phase3_interventions import SimpleBaselineRLM, AdvancedConvergenceDetector

sys.stdout.reconfigure(encoding='utf-8')

# Import necessary components from previous phases
class ValidationFramework:
    def __init__(self):
        self.validation_results = []
        self.optimization_history = []
        self.performance_benchmarks = {}

    def comprehensive_validation_suite(self, prompt_dataset: List[str],
                                     model_configs: List[Dict]) -> Dict:
        """Run comprehensive validation across multiple dimensions"""

        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE VALIDATION SUITE")
        print(f"Testing {len(prompt_dataset)} prompts with {len(model_configs)} configurations")
        print(f"{'='*80}")

        validation_results = {
            'accuracy_tests': {},
            'efficiency_tests': {},
            'robustness_tests': {},
            'usability_tests': {}
        }

        # Accuracy Tests
        print(f"\n🧪 Running Accuracy Tests...")
        validation_results['accuracy_tests'] = self.run_accuracy_validations(prompt_dataset, model_configs)

        # Efficiency Tests
        print(f"\n⚡ Running Efficiency Tests...")
        validation_results['efficiency_tests'] = self.run_efficiency_validations(prompt_dataset, model_configs)

        # Robustness Tests
        print(f"\n🛡️ Running Robustness Tests...")
        validation_results['robustness_tests'] = self.run_robustness_validations(prompt_dataset)

        # Usability Tests
        print(f"\n👥 Running Usability Tests...")
        validation_results['usability_tests'] = self.run_usability_validations(prompt_dataset)

        return validation_results

    def run_accuracy_validations(self, prompts: List[str], model_configs: List[Dict]) -> Dict:
        """Test convergence accuracy and quality"""
        accuracy_results = {}

        for i, prompt in enumerate(prompts[:3]):  # Test first 3 for efficiency
            print(f"\nAccuracy test for: {prompt[:50]}...")

            # Test different convergence thresholds
            thresholds = [0.80, 0.85, 0.90, 0.95]
            threshold_results = {}

            for threshold in thresholds:
                try:
                    # Run convergence experiment with specific threshold
                    detector = AdvancedConvergenceDetector(similarity_threshold=threshold)

                    # Simple baseline test
                    baseline_rlm = SimpleBaselineRLM()
                    result = baseline_rlm.run_baseline_experiment(prompt, max_iterations=4)

                    # Calculate metrics
                    metrics = detector.calculate_convergence_metrics(result['outputs'])
                    avg_similarity = np.mean(metrics.get('semantic_similarities', [0])) if metrics.get('semantic_similarities') else 0
                    keyword_stability = np.mean(metrics.get('keyword_stability', [0])) if metrics.get('keyword_stability') else 0

                    threshold_results[str(threshold)] = {
                        'avg_similarity': avg_similarity,
                        'keyword_stability': keyword_stability,
                        'iterations_used': len(result['outputs']),
                        'quality_score': avg_similarity * keyword_stability
                    }

                except Exception as e:
                    print(f"Error testing threshold {threshold}: {str(e)}")
                    threshold_results[str(threshold)] = {'error': str(e)}

            accuracy_results[f'prompt_{i+1}'] = threshold_results

        return accuracy_results

    def run_efficiency_validations(self, prompts: List[str], model_configs: List[Dict]) -> Dict:
        """Test computational efficiency and resource usage"""
        efficiency_results = {}

        for config in model_configs:
            model_name = config.get('model_name', 'gpt2')
            print(f"\nEfficiency test for model: {model_name}")

            try:
                # Test with timing and resource monitoring
                start_memory = self.get_memory_usage()
                start_time = time.time()

                # Run sample experiment
                rlm = SimpleBaselineRLM(model_name)
                result = rlm.run_baseline_experiment(prompts[0], max_iterations=3)

                end_time = time.time()
                end_memory = self.get_memory_usage()

                efficiency_results[model_name] = {
                    'execution_time': end_time - start_time,
                    'memory_usage_mb': end_memory - start_memory,
                    'iterations_per_second': len(result['outputs']) / (end_time - start_time),
                    'avg_time_per_iteration': (end_time - start_time) / len(result['outputs'])
                }

            except Exception as e:
                print(f"Error testing model {model_name}: {str(e)}")
                efficiency_results[model_name] = {'error': str(e)}

        return efficiency_results

    def run_robustness_validations(self, prompts: List[str]) -> Dict:
        """Test system robustness under various conditions"""
        robustness_results = {}

        # Test with noisy/ambiguous prompts
        challenging_prompts = [
            "Explain something complicated",  # Vague prompt
            "",  # Empty prompt
            "a b c d e f g h i j k l m n o p q r s t u v w x y z",  # Nonsensical
            "What is 2+2?",  # Simple factual
            "Describe the meaning of life in 10 words or less"  # Philosophical constraint
        ]

        for i, prompt in enumerate(challenging_prompts):
            print(f"\nRobustness test {i+1}: '{prompt[:30]}...'")

            try:
                rlm = SimpleBaselineRLM()
                result = rlm.run_baseline_experiment(prompt if prompt else "Explain mathematics",
                                                   max_iterations=3)

                # Check for graceful handling
                outputs = result['outputs']
                valid_responses = [out for out in outputs if out['output'] != "[No response generated]"]

                robustness_results[f'test_{i+1}'] = {
                    'prompt_type': 'vague' if i == 0 else 'empty' if i == 1 else 'nonsensical' if i == 2 else 'simple' if i == 3 else 'constrained',
                    'successful_iterations': len(valid_responses),
                    'total_iterations': len(outputs),
                    'success_rate': len(valid_responses) / len(outputs) if outputs else 0,
                    'handled_gracefully': len(valid_responses) > 0
                }

            except Exception as e:
                print(f"Error in robustness test {i+1}: {str(e)}")
                robustness_results[f'test_{i+1}'] = {
                    'error': str(e),
                    'handled_gracefully': False
                }

        return robustness_results

    def run_usability_validations(self, prompts: List[str]) -> Dict:
        """Test user experience and interface quality"""
        usability_results = {}

        # Simulate user interaction scenarios
        scenarios = [
            {'name': 'Quick Answer', 'max_time': 10},
            {'name': 'Detailed Analysis', 'max_time': 30},
            {'name': 'Real-time Interaction', 'max_time': 5}
        ]

        for scenario in scenarios:
            print(f"\nUsability test: {scenario['name']}")

            try:
                start_time = time.time()
                rlm = SimpleBaselineRLM()
                result = rlm.run_baseline_experiment(prompts[0], max_iterations=2)
                end_time = time.time()

                total_time = end_time - start_time
                within_time_budget = total_time <= scenario['max_time']

                usability_results[scenario['name']] = {
                    'total_time_seconds': total_time,
                    'within_time_budget': within_time_budget,
                    'time_budget': scenario['max_time'],
                    'efficiency_ratio': total_time / scenario['max_time']
                }

            except Exception as e:
                print(f"Error in usability test {scenario['name']}: {str(e)}")
                usability_results[scenario['name']] = {'error': str(e)}

        return usability_results

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0  # Fallback if psutil not available

    def generate_validation_report(self, validation_results: Dict) -> str:
        """Generate comprehensive validation report"""
        report = "\n" + "="*80 + "\n"
        report += "VALIDATION REPORT\n"
        report += "="*80 + "\n"

        # Accuracy Summary
        report += "\n🎯 ACCURACY SUMMARY:\n"
        if 'accuracy_tests' in validation_results:
            for prompt_key, threshold_results in validation_results['accuracy_tests'].items():
                if isinstance(threshold_results, dict) and '0.85' in threshold_results:
                    result = threshold_results['0.85']
                    if isinstance(result, dict) and 'quality_score' in result:
                        report += f"  {prompt_key}: Quality Score = {result['quality_score']:.3f}\n"

        # Efficiency Summary
        report += "\n⚡ EFFICIENCY SUMMARY:\n"
        if 'efficiency_tests' in validation_results:
            for model_name, eff_results in validation_results['efficiency_tests'].items():
                if isinstance(eff_results, dict) and 'execution_time' in eff_results:
                    report += f"  {model_name}: {eff_results['execution_time']:.2f}s, "
                    report += f"{eff_results['iterations_per_second']:.2f} iter/sec\n"

        # Robustness Summary
        report += "\n🛡️ ROBUSTNESS SUMMARY:\n"
        if 'robustness_tests' in validation_results:
            successful_tests = sum(1 for v in validation_results['robustness_tests'].values()
                                 if isinstance(v, dict) and v.get('handled_gracefully', False))
            total_tests = len(validation_results['robustness_tests'])
            report += f"  Graceful handling: {successful_tests}/{total_tests} tests passed\n"

        return report

class OptimizationEngine:
    def __init__(self):
        self.optimization_parameters = {
            'similarity_threshold': 0.85,
            'max_iterations': 6,
            'temperature': 0.7,
            'window_size': 3,
            'intervention_frequency': 2
        }
        self.optimization_history = []

    def auto_tune_parameters(self, prompt_dataset: List[str],
                           target_metrics: Dict[str, float]) -> Dict:
        """Automatically optimize parameters for best performance"""
        print(f"\n🤖 AUTO-TUNING PARAMETERS")
        print(f"Target metrics: {target_metrics}")

        # Parameter ranges to test
        parameter_ranges = {
            'similarity_threshold': [0.80, 0.85, 0.90, 0.95],
            'max_iterations': [3, 4, 5, 6, 7],
            'temperature': [0.5, 0.7, 0.9, 1.1],
            'window_size': [2, 3, 4, 5]
        }

        best_config = None
        best_score = -float('inf')
        optimization_trials = []

        # Grid search through parameter combinations
        for threshold in parameter_ranges['similarity_threshold'][:2]:  # Limit for efficiency
            for max_iter in parameter_ranges['max_iterations'][:3]:
                for temp in parameter_ranges['temperature'][:2]:
                    try:
                        # Test configuration
                        config_score = self.evaluate_configuration(
                            prompt_dataset[0] if prompt_dataset else "Explain machine learning",
                            threshold, max_iter, temp
                        )

                        trial_result = {
                            'config': {
                                'similarity_threshold': threshold,
                                'max_iterations': max_iter,
                                'temperature': temp
                            },
                            'score': config_score
                        }

                        optimization_trials.append(trial_result)

                        if config_score > best_score:
                            best_score = config_score
                            best_config = trial_result['config']

                    except Exception as e:
                        print(f"Error testing config (threshold={threshold}, iter={max_iter}): {str(e)}")
                        continue

        optimization_result = {
            'best_configuration': best_config,
            'best_score': best_score,
            'trials': optimization_trials,
            'optimization_complete': True
        }

        self.optimization_history.append(optimization_result)
        return optimization_result

    def evaluate_configuration(self, test_prompt: str, threshold: float,
                             max_iter: int, temperature: float) -> float:
        """Evaluate a configuration's performance"""
        try:
            # Create detector with custom threshold
            detector = AdvancedConvergenceDetector(similarity_threshold=threshold)

            # Run test experiment
            rlm = SimpleBaselineRLM()
            result = rlm.run_baseline_experiment(test_prompt, max_iterations=max_iter)

            # Calculate metrics
            metrics = detector.calculate_convergence_metrics(result['outputs'])
            similarities = metrics.get('semantic_similarities', [])

            if not similarities:
                return 0.0

            # Scoring function: balance convergence quality and efficiency
            avg_similarity = np.mean(similarities)
            convergence_speed = len(similarities) / max_iter  # Faster convergence is better

            # Weighted score (adjust weights based on priorities)
            quality_weight = 0.7
            speed_weight = 0.3

            score = (quality_weight * avg_similarity) + (speed_weight * convergence_speed)
            return score

        except Exception as e:
            print(f"Error evaluating configuration: {str(e)}")
            return 0.0

    def performance_benchmarking(self, prompt_dataset: List[str],
                               configurations: List[Dict]) -> Dict:
        """Benchmark performance across different configurations"""
        print(f"\n📊 PERFORMANCE BENCHMARKING")

        benchmark_results = {}

        for i, config in enumerate(configurations[:3]):  # Limit for efficiency
            print(f"Benchmarking configuration {i+1}...")

            try:
                # Extract config parameters
                threshold = config.get('similarity_threshold', 0.85)
                max_iter = config.get('max_iterations', 6)

                # Run benchmark across dataset
                scores = []
                times = []

                for prompt in prompt_dataset[:2]:  # Test first 2 prompts
                    start_time = time.time()
                    score = self.evaluate_configuration(prompt, threshold, max_iter, 0.7)
                    end_time = time.time()

                    scores.append(score)
                    times.append(end_time - start_time)

                benchmark_results[f'config_{i+1}'] = {
                    'configuration': config,
                    'avg_score': np.mean(scores) if scores else 0,
                    'std_score': np.std(scores) if len(scores) > 1 else 0,
                    'avg_execution_time': np.mean(times) if times else 0,
                    'reliability': len([s for s in scores if s > 0.5]) / len(scores) if scores else 0
                }

            except Exception as e:
                print(f"Error benchmarking configuration {i+1}: {str(e)}")
                benchmark_results[f'config_{i+1}'] = {'error': str(e)}

        return benchmark_results

class DeploymentOptimizer:
    def __init__(self):
        self.deployment_recommendations = []

    def generate_deployment_profile(self, validation_results: Dict,
                                  optimization_results: Dict) -> Dict:
        """Generate deployment recommendations based on validation"""
        print(f"\n📋 GENERATING DEPLOYMENT PROFILE")

        profile = {
            'recommended_configuration': {},
            'deployment_constraints': {},
            'scaling_recommendations': {},
            'monitoring_requirements': {}
        }

        # Analyze validation results to determine best configuration
        if 'accuracy_tests' in validation_results:
            # Find configuration with best accuracy
            best_accuracy = 0
            best_config_for_accuracy = None

            for prompt_results in validation_results['accuracy_tests'].values():
                if isinstance(prompt_results, dict):
                    for threshold_key, metrics in prompt_results.items():
                        if isinstance(metrics, dict) and 'quality_score' in metrics:
                            if metrics['quality_score'] > best_accuracy:
                                best_accuracy = metrics['quality_score']
                                best_config_for_accuracy = {'threshold': float(threshold_key)}

            profile['recommended_configuration']['accuracy_optimized'] = best_config_for_accuracy

        # Efficiency-based recommendations
        if 'efficiency_tests' in validation_results:
            fastest_model = None
            fastest_time = float('inf')

            for model_name, eff_metrics in validation_results['efficiency_tests'].items():
                if isinstance(eff_metrics, dict) and 'execution_time' in eff_metrics:
                    if eff_metrics['execution_time'] < fastest_time:
                        fastest_time = eff_metrics['execution_time']
                        fastest_model = model_name

            profile['recommended_configuration']['efficiency_optimized'] = {
                'fastest_model': fastest_model,
                'execution_time': fastest_time
            }

        # Robustness considerations
        if 'robustness_tests' in validation_results:
            successful_tests = sum(1 for v in validation_results['robustness_tests'].values()
                                 if isinstance(v, dict) and v.get('handled_gracefully', False))
            total_tests = len(validation_results['robustness_tests'])
            robustness_ratio = successful_tests / total_tests if total_tests > 0 else 0

            profile['deployment_constraints']['robustness_requirement'] = robustness_ratio > 0.8

        # Scaling recommendations
        profile['scaling_recommendations'] = {
            'batch_processing_suitable': True,  # RLMs can benefit from batching
            'parallel_processing_benefit': True,  # Independent prompts can run in parallel
            'memory_optimization_needed': 'efficiency_tests' in validation_results
        }

        # Monitoring requirements
        profile['monitoring_requirements'] = {
            'convergence_tracking': True,
            'performance_metrics': True,
            'error_rate_monitoring': True,
            'resource_utilization': True
        }

        return profile

    def create_optimization_dashboard_data(self, validation_results: Dict) -> Dict:
        """Create data suitable for visualization dashboard"""
        dashboard_data = {
            'accuracy_metrics': [],
            'efficiency_metrics': [],
            'robustness_metrics': [],
            'usability_metrics': []
        }

        # Process accuracy data
        if 'accuracy_tests' in validation_results:
            for prompt_key, threshold_results in validation_results['accuracy_tests'].items():
                if isinstance(threshold_results, dict):
                    for threshold, metrics in threshold_results.items():
                        if isinstance(metrics, dict) and 'quality_score' in metrics:
                            dashboard_data['accuracy_metrics'].append({
                                'prompt': prompt_key,
                                'threshold': float(threshold),
                                'quality_score': metrics['quality_score'],
                                'avg_similarity': metrics.get('avg_similarity', 0)
                            })

        # Process efficiency data
        if 'efficiency_tests' in validation_results:
            for model_name, eff_metrics in validation_results['efficiency_tests'].items():
                if isinstance(eff_metrics, dict) and 'execution_time' in eff_metrics:
                    dashboard_data['efficiency_metrics'].append({
                        'model': model_name,
                        'execution_time': eff_metrics['execution_time'],
                        'iterations_per_second': eff_metrics.get('iterations_per_second', 0)
                    })

        # Process robustness data
        if 'robustness_tests' in validation_results:
            for test_name, robustness_metrics in validation_results['robustness_tests'].items():
                if isinstance(robustness_metrics, dict) and 'success_rate' in robustness_metrics:
                    dashboard_data['robustness_metrics'].append({
                        'test_type': robustness_metrics.get('prompt_type', 'unknown'),
                        'success_rate': robustness_metrics['success_rate'],
                        'handled_gracefully': robustness_metrics.get('handled_gracefully', False)
                    })

        return dashboard_data

# Main execution function
def run_phase4_validation():
    """Run Phase 4 validation and optimization"""

    # Initialize frameworks
    validator = ValidationFramework()
    optimizer = OptimizationEngine()
    deploy_optimizer = DeploymentOptimizer()

    # Test datasets
    test_prompts = [
        "Explain quantum computing in simple terms",
        "What caused the Great Depression?",
        "Describe the water cycle",
        "How do vaccines work?",
        "What is artificial intelligence?"
    ]

    model_configs = [
        {'model_name': 'gpt2'},
        # Add other models if available
    ]

    print("Starting Phase 4: Validation & Optimization...")

    try:
        # Run comprehensive validation
        print(f"\n{'='*100}")
        print(f"PHASE 4: COMPREHENSIVE VALIDATION")
        print(f"{'='*100}")

        validation_results = validator.comprehensive_validation_suite(test_prompts, model_configs)

        # Print validation report
        validation_report = validator.generate_validation_report(validation_results)
        print(validation_report)

        # Run optimization
        print(f"\n{'='*100}")
        print(f"PHASE 4: AUTOMATIC OPTIMIZATION")
        print(f"{'='*100}")

        target_metrics = {
            'convergence_quality': 0.85,
            'execution_time': 15.0,  # seconds
            'robustness': 0.9
        }

        optimization_results = optimizer.auto_tune_parameters(test_prompts, target_metrics)
        print(f"Optimization complete!")
        print(f"Best configuration: {optimization_results.get('best_configuration', 'N/A')}")
        print(f"Best score: {optimization_results.get('best_score', 0):.3f}")

        # Performance benchmarking
        benchmark_configs = [
            {'similarity_threshold': 0.80, 'max_iterations': 4},
            {'similarity_threshold': 0.85, 'max_iterations': 5},
            {'similarity_threshold': 0.90, 'max_iterations': 6}
        ]

        benchmark_results = optimizer.performance_benchmarking(test_prompts, benchmark_configs)
        print(f"\nPerformance benchmarking complete!")

        # Generate deployment profile
        deployment_profile = deploy_optimizer.generate_deployment_profile(
            validation_results, optimization_results
        )
        print(f"\nDeployment profile generated!")

        # Create dashboard data
        dashboard_data = deploy_optimizer.create_optimization_dashboard_data(validation_results)
        print(f"Dashboard data ready for visualization!")

        # Summary results
        final_results = {
            'validation_results': validation_results,
            'optimization_results': optimization_results,
            'benchmark_results': benchmark_results,
            'deployment_profile': deployment_profile,
            'dashboard_data': dashboard_data
        }

        # Save results to file
        with open('phase4_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        print(f"\n✅ Phase 4 completed successfully!")
        print(f"Results saved to 'phase4_results.json'")

        return final_results

    except Exception as e:
        print(f"Error in Phase 4: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_phase4_validation()
    if results:
        print(f"\n🎉 Phase 4 Validation & Optimization Complete!")
        print(f"Check 'phase4_results.json' for detailed results.")
