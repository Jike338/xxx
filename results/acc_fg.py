import json
from collections import defaultdict

def main():
    input_path = "/home/jikezhong/Video-R1/results/7b_muma_sft_mmtom_questions_text_only.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    results = defaultdict(lambda: {"correct": 0, "total": 0})

    total_correct = 0
    total_questions = 0

    for q in data["questions"]:
        qtype = q["question_type"]
        reward = q["reward"]
        results[qtype]["correct"] += reward
        results[qtype]["total"] += 1

        total_correct += reward
        total_questions += 1

    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    print("Accuracy by question_type:")
    print(f"  TOTAL: {overall_accuracy:.3f} ({int(total_correct)} / {total_questions})")

    for qtype in sorted(results):
        correct = results[qtype]["correct"]
        total = results[qtype]["total"]
        accuracy = correct / total if total > 0 else 0.0
        print(f"  Question Type {qtype:.1f}: {accuracy:.3f} ({int(correct)} / {total})")

if __name__ == "__main__":
    main()
