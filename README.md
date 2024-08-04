# SAE Chatbot Evolution

This project uses Sparse Autoencoders (SAE) to evolve chatbots through a process of genetic algorithms and natural language processing. It leverages the Groq API for language model interactions and implements custom evolution strategies to optimize chatbot behavior based on specified criteria.

## Features

- Evolutionary algorithm for chatbot optimization
- Integration with Groq API for language model interactions
- Customizable criteria for chatbot evaluation
- Population-based approach with elite selection
- Configurable evolution parameters

## Prerequisites

- Python 3.7+
- Groq API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sae-chatbot-evolution.git
   cd sae-chatbot-evolution
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Groq API key as an environment variable:
   ```
   export GROQ_API_KEY=your_api_key_here
   ```

## Usage

Run the main script `sae_search.py` with desired parameters. Here's an example:

```
python sae_search.py --cycles 10 --elite 5 --population 20 --initial-population 10 --criteria examples/happy-chatbot.yaml --model llama3-70b-8192 --coeff-start 30 --coeff-end 120
```

### Parameters

- `--cycles`: Number of evolution cycles (default: 10)
- `--elite`: Number of elite candidates to preserve (default: 5)
- `--population`: Total population size (default: 15)
- `--initial-population`: Initial population size (default: 2)
- `--criteria`: YAML file containing evolution criteria (default: "examples/sports_coach.yaml")
- `--model`: Groq model to use (default: "llama-3.1-70b-versatile")
- `--coeff-start`: Start of coefficient range (default: 40)
- `--coeff-end`: End of coefficient range (default: 200)
- `--seed`: Random seed for reproducibility (optional)

## Project Structure

- `sae_search.py`: Main script for running the evolution process
- `sae_evolution.py`: Contains evolution-related functions (Candidate, breed, mutation, crossover, etc.)
- `examples/`: Directory containing sample criteria YAML files
- `results/`: Directory where evolution results are stored

## How It Works

1. The script initializes a population of chatbot candidates with random SAE configurations.
2. For each evolution cycle:
   - Candidates are compared based on the specified criteria using the Groq API.
   - The best-performing candidates are selected as elites.
   - New candidates are generated through breeding, mutation, and crossover.
3. The process repeats for the specified number of cycles.
4. The final population is saved, and the best candidate is rendered in HTML.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Acknowledgements

- Groq for providing the API for language model interactions
- [Any other acknowledgements]