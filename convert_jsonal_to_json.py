import json

in_file = "/home/jikezhong/Video-R1/mmtom_questions.jsonl"
out_file = "/home/jikezhong/Video-R1/mmtom_questions.json"
# Load each line as a JSON object
with open(in_file, "r") as f:
    data = [json.loads(line) for line in f if line.strip()]

# Save as a valid JSON array
with open(out_file, "w") as f:
    json.dump(data, f, indent=2)
