import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import sys

# Setting encoding explicitly for standard output due to Unicode errors during pipelining on Windows
sys.stdout.reconfigure(encoding='utf-8')

class ConvergenceRLM:
    def __init__(self, model_name="gpt2"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

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

    def calculate_semantic_similarity(self, text1, text2):
        """Calculate cosine similarity between sentence embeddings"""
        if not text1.strip() or not text2.strip():
            return 0.0

        embeddings = self.similarity_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def calculate_syntactic_similarity(self, text1, text2):
        """Simple character-level similarity using edit distance"""
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        if not text1 or not text2:
            return 0.0

        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0

        distance = levenshtein_distance(text1.lower(), text2.lower())
        similarity = 1 - (distance / max_len)
        return similarity

def run_convergence_experiment(rlm, initial_prompt, max_iterations=5, semantic_threshold=0.85):
    """
    Run convergence experiment and track metrics
    """
    print(f"\n{'='*60}")
    print(f"CONVERGENCE EXPERIMENT")
    print(f"Initial Prompt: {initial_prompt}")
    print(f"{'='*60}")

    outputs = []
    semantic_similarities = []
    syntactic_similarities = []

    # First iteration
    current_output = rlm.generate_response(initial_prompt)
    outputs.append({
        'iteration': 1,
        'prompt': initial_prompt,
        'output': current_output
    })

    print(f"\nIteration 1:")
    print(f"Output: {current_output}")

    # Subsequent iterations
    for i in range(1, max_iterations):
        # Create reflective prompt
        reflective_prompt = f"""
Original question: {initial_prompt}
Previous answer: {current_output}

Please improve and refine the answer above. Make it more accurate and complete:"""

        new_output = rlm.generate_response(reflective_prompt)
        outputs.append({
            'iteration': i+1,
            'prompt': reflective_prompt,
            'output': new_output
        })

        # Calculate similarities
        semantic_sim = rlm.calculate_semantic_similarity(current_output, new_output)
        syntactic_sim = rlm.calculate_syntactic_similarity(current_output, new_output)

        semantic_similarities.append(semantic_sim)
        syntactic_similarities.append(syntactic_sim)

        print(f"\nIteration {i+1}:")
        print(f"Output: {new_output}")
        print(f"Semantic Similarity: {semantic_sim:.3f}")
        print(f"Syntactic Similarity: {syntactic_sim:.3f}")

        # Check for convergence
        if semantic_sim > semantic_threshold:
            print(f"✅ CONVERGED at iteration {i+1}!")
            break

        current_output = new_output

    # Summary statistics
    if semantic_similarities:
        avg_semantic = np.mean(semantic_similarities)
        max_semantic = np.max(semantic_similarities)
        print(f"\nSUMMARY:")
        print(f"Average Semantic Similarity: {avg_semantic:.3f}")
        print(f"Maximum Semantic Similarity: {max_semantic:.3f}")
        print(f"Total Iterations: {len(outputs)}")

    return {
        'outputs': outputs,
        'semantic_similarities': semantic_similarities,
        'syntactic_similarities': syntactic_similarities
    }

# Test benchmark prompts
BENCHMARK_PROMPTS = [
    "What year did World War II end and what were the major consequences?",
    "Explain how photosynthesis works in simple terms.",
    "Write a short story about a robot learning to feel emotions.",
    "What are the main arguments for and against remote work?"
]

if __name__ == "__main__":
    # Initialize RLM
    rlm = ConvergenceRLM("gpt2")  # Start with smaller model

    # Run experiments on benchmark prompts
    results = {}

    for i, prompt in enumerate(BENCHMARK_PROMPTS):  # Run all 4 prompts
        print(f"\n{'#'*80}")
        print(f"BENCHMARK {i+1}/{len(BENCHMARK_PROMPTS)}")
        print(f"{'#'*80}")

        result = run_convergence_experiment(rlm, prompt, max_iterations=4)
        results[f'benchmark_{i+1}'] = result
        # Removed the input() blocker for automatic script testing
