import argparse
import asyncio
import logging
import random
import yaml
from pathlib import Path
from gemma.model_utils import process_prompt, steer_generate, sae_params
import groq
import os

# Import your evolution functions here
from sae_evolution import (
    Candidate, load_candidate, run_evolution, breed, mutation, crossover
)
api_key = os.getenv('GROQ_API_KEY')

if api_key is None:
    print("Error: GROQ_API_KEY environment variable is not set")
    sys.exit(1)

# Initialize the Groq client with the API key
client = groq.Client(api_key=api_key)


candidate_cache = {}

def generate_content(candidate, prefix):
    global candidate_cache
    if candidate in candidate_cache:
        return candidate_cache[candidate]
    candidate_cache[candidate] = "[Sample]:\n"+steer_generate(prefix, candidate.layers)+"\n[Sample]:\n"+steer_generate(prefix, candidate.layers)
    candidate.last_gen=candidate_cache[candidate]
    return candidate_cache[candidate]

async def compare_candidates(candidate1, candidate2, criteria, output_prefix, model):
    print("Generating candidate 1", candidate1.layers)
    gen1 = generate_content(candidate1, output_prefix)
    print("Generating candidate 2", candidate2.layers)
    gen2 = generate_content(candidate2, output_prefix)
    prompt = (f"Comparing candidates based on: {criteria}")
    prompt += (f"\nCandidate 1: ```\n{gen1}\n```")
    prompt += (f"\nCandidate 2: ```\n{gen2}\n```")
    #TODO
    prompt += "\nEnter 1 if Candidate 1 better matches the criteria, else 2 if Candidate 2 better matches the critera. Only output 1 or 2, this is automated.: "
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        max_tokens=128
    )
    result = chat_completion.choices[0].message.content
    print("--", result)

    return 1 if result == "1" else -1

async def main(args):
    logging.basicConfig(level=logging.INFO)
    
    if args.seed:
        random.seed(args.seed)
    # Load criteria file
    with open(args.criteria, 'r') as f:
        criteria_data = yaml.safe_load(f)
        initial_prompts = criteria_data['initial_prompts']
        criteria = criteria_data['criteria']
        output_prefix = criteria_data['output_prefix']

    # Create initial population TODO
    population = []
    for i in range(args.initial_population):
        prompt = random.choice(initial_prompts)

        #TODO
        target_layer = random.choice(list(sae_params().keys()))
        _, indices = process_prompt(prompt, target_layer)
        indices = random.sample(indices[0].tolist(), 3)
        indices_map = {key: random.randint(args.coeff_start, args.coeff_end) for key in indices}

        layers = {target_layer: indices_map}
        file_path = f"model_{i}.yaml" #TODO save
        candidate = Candidate(file_path, layers, initial_population=True)
        population.append(candidate)
    
    for cycle in range(args.cycles):
        logging.info(f"Starting cycle {cycle + 1}")
        population = await run_evolution(
            population,
            args.elite,
            args.population,
            0.1,  # mutation_rate
            lambda c1, c2: compare_candidates(c1, c2, criteria, output_prefix, args.model)
        )
        
        out_population = [candidate.to_dict() for candidate in population]
        with open(f"output-{cycle}.yaml", 'w') as f:
            yaml.dump(out_population, f)
    
    logging.info("Evolution complete. Final population:")
    for candidate in population:
        logging.info(f"Model: {candidate.file_path}, Generation: {candidate.generation}")
    
    # Save final population to YAML
    final_population = [candidate.to_dict() for candidate in population]
    with open('final_population.yaml', 'w') as f:
        yaml.dump(final_population, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAE evolution")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--cycles", type=int, default=10, help="Number of evolution cycles")
    parser.add_argument("--elite", type=int, default=5, help="Number of elite candidates")
    parser.add_argument("--coeff-start", type=int, default=40, help="Start of coefficient range")
    parser.add_argument("--coeff-end", type=int, default=200, help="End of coefficient range")
    parser.add_argument("--population", type=int, default=15, help="Population size")
    parser.add_argument("--initial-population", type=int, default=2, help="Initial population size")
    parser.add_argument("--criteria", type=str, default="examples/sports_coach.yaml", help="yml file created from metaprompt.py. See examples")
    parser.add_argument("--model", type=str, default="llama-3.1-70b-versatile", help="Which groq model to use")

    args = parser.parse_args()
    asyncio.run(main(args))
