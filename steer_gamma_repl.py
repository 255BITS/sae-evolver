import sys
import os
import yaml
from gemma.model_utils import steer_generate
from sae_evolution import load_candidate

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_prompt.py <yaml_file>")
        sys.exit(1)

    yaml_file = sys.argv[1]

    data = load_candidate(yaml_file)
    while(True):
        user_input = input("Enter your input: ")

        # Load candidate layers from the YAML file
        layers = {key: value for key, value in data.layers.items() if isinstance(value, dict)}
        
        
        output = steer_generate(user_input, {})
        print("Generated output without steering:")
        print('--', layers)
        print(output)

        output = steer_generate(user_input, layers)
        
        print("Generated output with steering:")
        print(output)

if __name__ == "__main__":
    main()

