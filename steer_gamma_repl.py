import sys
import os
import yaml
from gemma.model_utils import steer_generate

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_prompt.py <yaml_file>")
        sys.exit(1)

    yaml_file = sys.argv[1]

    if not os.path.exists(yaml_file):
        print(f"Error: File {yaml_file} does not exist")
        sys.exit(1)

    data = load_yaml(yaml_file)
    while(True):
        user_input = input("Enter your input: ")

        # Load candidate layers from the YAML file
        layers = {key: value for key, value in data.items() if isinstance(value, dict)}
        
        print("Candidate Layers:")
        print(layers)
        
        output = steer_generate(prompt, layers)
        
        print("Generated Output:")
        print(output)

if __name__ == "__main__":
    main()

