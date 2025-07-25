import json

def save_first_n_items(input_path, output_path, n=200):
    with open(input_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        trimmed = dict(list(data.items())[:n])
    elif isinstance(data, list):
        trimmed = data[:n]
    else:
        raise ValueError("Unsupported JSON top-level type")

    with open(output_path, 'w') as f:
        json.dump(trimmed, f, indent=2)

    print(f"Saved first {n} items from {input_path} to {output_path}")

# File paths
files = [
    "/home/jikezhong/Video-R1/Video-R1-260k.json",
    "/home/jikezhong/Video-R1/Video-R1-COT-165k.json"
]

for file in files:
    output_file = file.replace(".json", "-200.json")
    save_first_n_items(file, output_file)
