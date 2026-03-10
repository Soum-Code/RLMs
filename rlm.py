import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import numpy as np
import time

class SimpleRLM:
    def __init__(self, model_name="gpt2"):
        print(f"Loading models ({model_name} and all-MiniLM-L6-v2)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Models loaded successfully.")

    def generate_response(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        # Using sampling to get some variety, but greedy works too
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # We want to extract only the generated part
        generated_text = response[len(prompt):].strip()
        # Fallback if generation is wonky
        return generated_text if generated_text else "No specific answer."

    def calculate_similarity(self, text1, text2):
        if not text1 or not text2: return 0.0
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
        
    def calculate_edit_distance(self, text1, text2):
        # Using difflib to get a ratio and converting to distance (1 - ratio)
        return 1.0 - difflib.SequenceMatcher(None, text1, text2).ratio()
        
    def calculate_jaccard_similarity(self, text1, text2):
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 and not tokens2:
            return 1.0
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union) if len(union) > 0 else 0.0

def test_convergence(rlm, initial_prompt, max_iterations=5, threshold=0.85):
    outputs = []
    metrics = {
        "semantic_similarity": [],
        "edit_distance": [],
        "jaccard_similarity": [],
        "lengths": []
    }

    start_time = time.time()
    current_output = rlm.generate_response(initial_prompt)
    outputs.append(current_output)
    metrics["lengths"].append(len(current_output.split()))
    
    print(f"\nPrompt: '{initial_prompt}'")
    print(f"Iteration 1: {current_output}")

    converged_at = -1

    for i in range(1, max_iterations):
        # The prompt structure needs to encourage improvement or refinement
        new_prompt = f"Previous answer: {current_output}\n\nImprove this answer:"
        new_output = rlm.generate_response(new_prompt)
        outputs.append(new_output)
        metrics["lengths"].append(len(new_output.split()))

        # Calculate metrics
        sem_sim = rlm.calculate_similarity(current_output, new_output)
        edit_dist = rlm.calculate_edit_distance(current_output, new_output)
        jac_sim = rlm.calculate_jaccard_similarity(current_output, new_output)
        
        metrics["semantic_similarity"].append(sem_sim)
        metrics["edit_distance"].append(edit_dist)
        metrics["jaccard_similarity"].append(jac_sim)

        print(f"Iteration {i+1}: {new_output}")
        print(f"Metrics -> Semantic Sim: {sem_sim:.3f}, Jaccard Sim: {jac_sim:.3f}, Edit Dist: {edit_dist:.3f}, Length: {metrics['lengths'][-1]}")

        if sem_sim > threshold and converged_at == -1:
            print(f"--- Converged at iteration {i+1}! ---")
            converged_at = i + 1
            # We purposely don't break immediately to see if it diverges again
            # Or if we want strict termination, we uncomment below:
            # break

        current_output = new_output

    end_time = time.time()
    
    # Calculate length variance
    length_variance = np.var(metrics["lengths"]) if len(metrics["lengths"]) > 0 else 0
    time_to_convergence = end_time - start_time
    
    summary = {
        "converged": converged_at != -1,
        "converged_at_iteration": converged_at,
        "mean_semantic_similarity": np.mean(metrics["semantic_similarity"]) if metrics["semantic_similarity"] else 0,
        "length_variance": length_variance,
        "time_taken_seconds": time_to_convergence
    }

    return outputs, metrics, summary

def run_benchmarks():
    rlm = SimpleRLM()
    
    prompts = {
        "Factual Q&A": "What year did World War II end?",
        "Reasoning Problem": "If John has 5 apples and eats 2, then buys 3 more, how many does he have?",
        "Creative Task": "Continue the story: The old wooden door creaked open, revealing a glowing portal...",
        "Opinion Generation": "What are the pros and cons of remote work?"
    }
    
    results = {}
    for task_type, prompt in prompts.items():
        print(f"\n{'='*50}")
        print(f"Running Task: {task_type}")
        print(f"{'='*50}")
        outputs, metrics, summary = test_convergence(rlm, prompt, max_iterations=6, threshold=0.85)
        results[task_type] = {
            "prompt": prompt,
            "outputs": outputs,
            "metrics": metrics,
            "summary": summary
        }
    
    print("\n\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    for task, data in results.items():
        s = data['summary']
        status = f"Converged at iter {s['converged_at_iteration']}" if s['converged'] else "Failed to converge"
        print(f"{task}: {status}")
        print(f"  - Mean Semantic Sim: {s['mean_semantic_similarity']:.3f}")
        print(f"  - Length Variance:   {s['length_variance']:.2f}")

if __name__ == "__main__":
    run_benchmarks()
