import requests
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import sys

# Setting encoding explicitly for standard output due to Unicode errors during pipelining on Windows
sys.stdout.reconfigure(encoding='utf-8')

class OllamaRLM:
    def __init__(self, model_name="gpt-oss:20b", host="http://localhost:11434"):
        print(f"Connecting to Ollama model: {model_name} at {host}")
        self.model_name = model_name
        self.api_url = f"{host}/api/generate"
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Models loaded successfully!")

    def generate_response(self, prompt, max_length=200, temperature=0.7):
        # We use the /api/generate endpoint for Ollama
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

    def calculate_semantic_similarity(self, text1, text2):
        if not text1.strip() or not text2.strip():
            return 0.0

        embeddings = self.similarity_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def calculate_syntactic_similarity(self, text1, text2):
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
    print(f"\n{'='*60}")
    print(f"CONVERGENCE EXPERIMENT")
    print(f"Initial Prompt: {initial_prompt}")
    print(f"{'='*60}")

    outputs = []
    semantic_similarities = []
    syntactic_similarities = []

    start_time = time.time()
    current_output = rlm.generate_response(initial_prompt)
    outputs.append({
        'iteration': 1,
        'prompt': initial_prompt,
        'output': current_output
    })

    print(f"\nIteration 1:")
    print(f"Output: {current_output}")

    for i in range(1, max_iterations):
        # We stick to the reflective prompt to see if the 20B OSS model can handle it
        # unlike the baseline 124M gpt2.
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

        semantic_sim = rlm.calculate_semantic_similarity(current_output, new_output)
        syntactic_sim = rlm.calculate_syntactic_similarity(current_output, new_output)

        semantic_similarities.append(semantic_sim)
        syntactic_similarities.append(syntactic_sim)

        print(f"\nIteration {i+1}:")
        print(f"Output: {new_output}")
        print(f"Semantic Similarity: {semantic_sim:.3f}")
        print(f"Syntactic Similarity: {syntactic_sim:.3f}")

        if semantic_sim > semantic_threshold:
            print(f"✅ CONVERGED at iteration {i+1}!")
            break

        current_output = new_output

    if semantic_similarities:
        avg_semantic = np.mean(semantic_similarities)
        print(f"\nSUMMARY:")
        print(f"Average Semantic Similarity: {avg_semantic:.3f}")
        print(f"Total Iterations: {len(outputs)}")
        print(f"Time Taken: {time.time() - start_time:.2f} seconds")

    return True

BENCHMARK_PROMPTS = [
    "What year did World War II end and what were the major consequences?",
    "Explain how photosynthesis works in simple terms.",
    "Write a short story about a robot learning to feel emotions.",
    "What are the main arguments for and against remote work?"
]

if __name__ == "__main__":
    # Ensure Ollama is running and has the model
    rlm = OllamaRLM("gpt-oss:20b")
    
    # Run the tests
    for i, prompt in enumerate(BENCHMARK_PROMPTS):
        print(f"\n{'#'*80}")
        print(f"BENCHMARK {i+1}/{len(BENCHMARK_PROMPTS)}")
        print(f"{'#'*80}")
        run_convergence_experiment(rlm, prompt, max_iterations=4)
