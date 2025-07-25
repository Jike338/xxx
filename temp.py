# import json
# from pathlib import Path

# # Path to your JSON file
# json_path = Path("/home/jikezhong/Video-R1/mmtom_questions.json")

# # Load the JSON content
# with open(json_path, "r") as f:
#     data = json.load(f)

# # Add question_id to each question
# for idx, item in enumerate(data):
#     item["question_id"] = idx

# # Write the updated data back to the file (backup first)
# backup_path = json_path.with_suffix(".json.bak")
# json_path.rename(backup_path)

# with open(json_path, "w") as f:
#     json.dump(data, f, indent=2)

# print(f"Updated file saved to: {json_path}")
# print(f"Original file backed up to: {backup_path}")

# import json

# # Path to the JSON file
# json_path = "/home/jikezhong/Video-R1/mmtom_questions.json"

# # Load the original data
# with open(json_path, "r") as f:
#     data = json.load(f)

# # Update each question with new fields
# for item in data:
#     qid = item["question_id"]
#     item["data_type"] = "video"
#     item["path"] = f"/temp/jz/mmtom_data/question_{qid}.mp4"

# # Save the updated data back to the original file
# with open(json_path, "w") as f:
#     json.dump(data, f, indent=2)

# import json

# # Input and output paths
# input_path = "/home/jikezhong/Video-R1/mmtom_questions.json"
# output_path = "/home/jikezhong/Video-R1/mmtom_questions_filtered.json"

# # Load the original JSON
# with open(input_path, "r") as f:
#     data = json.load(f)

# # Transform each item
# for item in data:
#     item["problem"] = item.pop("question")
#     item["problem_id"] = item.pop("question_id")
#     item["solution"] = item.pop("answer")
#     item["problem_type"] = "other"

# # Save the transformed data to a new file
# with open(output_path, "w") as f:
#     json.dump(data, f, indent=2)


import json
import re

def extract_question_text(problem_text):
    """Extracts the question part starting with 'Question:'"""
    match = re.search(r'Question:\s*(.*)', problem_text, re.DOTALL)
    return f"Question: {match.group(1).strip()}" if match else problem_text

def main(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if 'problem' in item:
            item['problem'] = extract_question_text(item['problem'])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Example usage
main('/home/jikezhong/Video-R1/mmtom_questions.json', '/home/jikezhong/Video-R1/mmtom_questions_vid_only.json')
