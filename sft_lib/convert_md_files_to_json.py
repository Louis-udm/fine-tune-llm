import json
from sft_lib.dataset_utils import get_dataset_from_text_files

# {"input":"<text>","output":"<text>"}


ANSWER_DELIMITER = "^^^^A^^^^"
ds = get_dataset_from_text_files("simple_markdown_with_answer", suffix="md")
json_ds = []
for sample in ds:
    text = sample["text"]
    answer_idx = text.find(ANSWER_DELIMITER)
    json_ds.append(
        {
            "input": text[:answer_idx],
            "output": text[answer_idx + len(ANSWER_DELIMITER) :],
        }
    )
# with open("datasets/simple_markdown_with_answer.json", "wt") as f:
#     for sample in json_ds:
#         f.write(json.dumps(sample) + "\n")

json.dump(json_ds, open("datasets/simple_markdown_with_answer.json", "wt"), indent=4)
