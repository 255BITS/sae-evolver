from gemma.model_utils import process_prompt, steer_generate

def main():
    prompt = "Would you be able to travel through time using a wormhole?"
    values, indices = process_prompt(prompt, 20)
    print("Values:", values)
    print("Indices:", indices)

    print(steer_generate("What is the most iconic structure known to man?", {20: {10004: 200.0}}))
    print(steer_generate("What is on your mind?", {20: {10004: 200.0}}))

if __name__ == "__main__":
    main()

