import os
import logging
import random
import torch
import uuid
import yaml
from pathlib import Path

class Candidate:
    def __init__(self, file_path, layers, initial_population=False, generation=0):
        self.file_path = file_path
        self.initial_population = initial_population
        self.layers = layers
        self.generation = generation
        self.last_gen = None

    def to_dict(self):
        return {
            "model": self.file_path,
            "generation": self.generation,
            "last_gen": self.last_gen,
            "layers": self.layers
        }

def crossover(parent1, parent2):
    child_layers = {}
    
    file_path = f"{uuid.uuid4()}.yaml"
    child = Candidate(file_path, child_layers)
    child.layers = crossover_layers(parent1.layers, parent2.layers)
    child.generation = max(parent1.generation, parent2.generation) + 1
    return child

def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key].update(value)
            else:
                result[key] = value.copy()
    return result

def random_filter(d, fraction=0.5):
    keys = list(d.keys())
    num_to_keep = max(1, int(len(keys) * fraction))
    keys_to_keep = random.sample(keys, num_to_keep)
    return {k: d[k] for k in keys_to_keep}

def crossover_layers(a, b):
    # Create initial dict with the first nested entry
    first_key = next(iter(a))
    first_nested_key = next(iter(a[first_key]))
    init = {first_key: {first_nested_key: a[first_key][first_nested_key]}}

    # Filter out half the nested entries in a and b randomly
    ahalf = {k: random_filter(v) for k, v in a.items()}
    bhalf = {k: random_filter(v) for k, v in b.items()}

    # Perform the crossover
    crossover_ab = merge_dicts(init, ahalf, bhalf)

    return crossover_ab

def mutation(candidate, mutation_rate=0.01, mutation_scale=20, rare_change_rate=0.001):
    mutated_layers = {}
    
    for layer_name, layer in candidate.layers.items():
        mutated_layer = {}
        for neuron_id, weight in layer.items():
            if random.random() < mutation_rate:
                # Randomize weight coefficient
                mutated_weight = weight + random.gauss(0, mutation_scale)
                mutated_layer[neuron_id] = int(mutated_weight)
            elif random.random() < rare_change_rate:
                # Rare chance to drop this neuron
                continue
            else:
                mutated_layer[neuron_id] = weight
        
        # Add a random neuron with a small chance
        if random.random() < rare_change_rate:
            new_neuron_id = max(layer.keys()) + 1 if layer else 0
            mutated_layer[new_neuron_id] = random.gauss(0, mutation_scale)
        
        if len(mutated_layer.items()) > 0:
            mutated_layers[layer_name] = mutated_layer
    
    candidate.layers = mutated_layers

def save_candidate(candidate, file_path):
    # Ensure the directory 'candidates' exists
    os.makedirs('candidates', exist_ok=True)
    
    # Construct the full file path
    full_path = os.path.join('candidates', file_path)
    
    # Save the candidate.layers map to the specified file in YAML format
    with open(full_path, 'w') as file:
        yaml.dump(candidate.to_dict(), file)

def load_candidate(file_name):
    # Construct the full file path
    full_path = os.path.join('candidates', file_name)
    
    # Load the YAML file into a dictionary
    with open(full_path, 'r') as file:
        candidate = yaml.safe_load(file)
    
    return Candidate(file_path=full_path, layers=candidate['layers'], initial_population=True)

def breed(parents, mutation_rate):
    print("breed", parents)
    offspring = crossover(parents[0], parents[1])
    mutation(offspring, mutation_rate)
    
    # Save the offspring's layers to a file
    save_candidate(offspring, offspring.file_path)
    
    return offspring

def selection(population):
    return random.sample(population, 2)

# You can keep the evolve function mostly the same, just update it to use the new breed function
def evolve(population, population_size, mutation_rate,):
    seed_population = list(population)
    while len(population) < population_size:
        parents = selection(seed_population)
        offspring = breed(parents, mutation_rate)
        population.append(offspring)

    return population

async def run_evolution(population, elite_size, population_size, mutation_rate, evaluation_criteria):
    logging.info("Before evolve")
    log_candidates(population)
    
    population = evolve(population, population_size, mutation_rate)

    logging.info("Before sorting")
    log_candidates(population)
    
    population = await sort_with_correction(population, evaluation_criteria)
    
    logging.info("After sorting")
    log_candidates(population)
    
    return population[:elite_size]

def log_candidates(population):
    format_str = "{0}. {1:<24}"
    for index, candidate in enumerate(population, start=1):
        logging.info(format_str.format(index, candidate.file_path))


async def correct_insert_element(item, sorted_list, compare, top_k):
    if not sorted_list:
        return [item]
    # find a place for insertion
    insert_pos = await find_insertion_point(item, sorted_list, compare, top_k)
    # insert item tentatively
    sorted_list.insert(insert_pos, item)
    return sorted_list

async def find_insertion_point(item, sorted_list, compare, top_k):
    # binary search variant that accounts for potential comparison errors
    low, high = 0, len(sorted_list) - 1
    while low <= high:
        if low > top_k and top_k > 0:
            return low
        mid = (low + high) // 2
        result = await compare(item, sorted_list[mid])
        # adjust binary search based on comparison, considering potential inaccuracies
        if result == 1:
            high = mid - 1
        else:
            low = mid + 1
    return low

async def sort_with_correction(buffer, compare, top_k=-1):
    sorted_list = []
    for item in buffer:
        sorted_list = await correct_insert_element(item, sorted_list, compare, top_k)
    # correction mechanism here
    #sorted_list = await correction_pass(sorted_list)
    return sorted_list
