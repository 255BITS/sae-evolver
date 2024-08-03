import argparse
import asyncio
import logging
import random
import yaml
from pathlib import Path
from gemma.model_utils import process_prompt, steer_generate

# Import your evolution functions here
from sae_evolution import (
    Candidate, load_candidate, run_evolution, breed, mutation, crossover
)

candidate_cache = {}

def generate_content(candidate, prefix):
    global candidate_cache
    if candidate in candidate_cache:
        return candidate_cache[candidate]
    candidate_cache[candidate] = steer_generate(prefix, candidate.layers)
    return candidate_cache[candidate]

async def compare_candidates(candidate1, candidate2, criteria, output_prefix):
    gen1 = generate_content(candidate1, output_prefix)
    gen2 = generate_content(candidate2, output_prefix)
    print(f"Comparing candidates based on: {criteria}")
    print(f"Candidate 1: ```\n{gen1}\n```")
    print(f"Candidate 2: ```\n{gen2}\n```")
    #TODO
    result = input("Enter 1 if Candidate 1 better matches the criteria, else 2 if Candidate 2 better matches the critera: ")
    return 1 if result == "1" else -1

async def main(args):
    logging.basicConfig(level=logging.INFO)
    
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
        target_layer = 20#random.randint(19, 26)
        _, indices = process_prompt(prompt, target_layer)
        indices = random.sample(indices[0].tolist(), 3)
        indices_map = {key: random.randint(50, 350) for key in indices}

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
            lambda c1, c2: compare_candidates(c1, c2, criteria, output_prefix)
        )
    
    logging.info("Evolution complete. Final population:")
    for candidate in population:
        logging.info(f"Model: {candidate.file_path}, Generation: {candidate.generation}")
    
    # Save final population to YAML
    final_population = [candidate.to_dict() for candidate in population]
    with open('final_population.yaml', 'w') as f:
        yaml.dump(final_population, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAE evolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cycles", type=int, default=10, help="Number of evolution cycles")
    parser.add_argument("--elite", type=int, default=5, help="Number of elite candidates")
    parser.add_argument("--population", type=int, default=15, help="Population size")
    parser.add_argument("--initial-population", type=int, default=2, help="Initial population size")
    parser.add_argument("--criteria", type=str, default="examples/sports_coach.yaml", help="yml file created from metaprompt.py. See examples")

    args = parser.parse_args()
    asyncio.run(main(args))
