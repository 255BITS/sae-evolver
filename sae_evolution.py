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

    def to_dict(self):
        return {
            "model": self.file_path,
            "generation": self.generation,
            "layers": self.layers
        }

def crossover(parent1, parent2):
    child_layers = {}
    
    file_path = f"{uuid.uuid4()}.yaml"
    child = Candidate(file_path, child_layers)
    child.generation = max(parent1.generation, parent2.generation) + 1
    return child

def mutation(candidate, mutation_rate=0.01, mutation_scale=0.01):
    mutated_layers = {}
    for layer_name, layer in candidate.layers.items():
        if random.random() < mutation_rate:
            mutation = torch.randn_like(layer) * mutation_scale
            mutated_layers[layer_name] = layer + mutation
        else:
            mutated_layers[layer_name] = layer.clone()
    
    candidate.layers = mutated_layers

def save_candidate(candidate, file_path):
    # Ensure the directory 'candidates' exists
    os.makedirs('candidates', exist_ok=True)
    
    # Construct the full file path
    full_path = os.path.join('candidates', file_path)
    
    # Save the candidate.layers map to the specified file in YAML format
    with open(full_path, 'w') as file:
        yaml.dump(candidate.layers, file)

def load_candidate(file_name):
    # Construct the full file path
    full_path = os.path.join('candidates', file_name)
    
    # Load the YAML file into a dictionary
    with open(full_path, 'r') as file:
        layers = yaml.safe_load(file)
    
    return Candidate(layers=layers, initial_population=True)

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
    
    for tokill in population[elite_size:]:
        if not tokill.initial_population:
            os.remove(tokill.file_path)
    
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
