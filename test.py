import os
import json
import torch
from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)


class Statement:
    def __init__(self, origin, line_nr, code_statement, vulnerable) -> None:
        self.origin = origin
        self.line_nr = line_nr
        self.code_statement = code_statement
        self.vulnerable = vulnerable
        self.tokens = tokenizer.encode(code_statement, return_tensors="pt")


dataset = []
root_dir = "./llm-vul/VJBench-trans"

# Re-model the dataset
# Create one object containing each line of code and if it is vulnerable or not
max_length = 0

for root, dirs, files in os.walk(root_dir):
    for dir in dirs:
        json_file = os.path.join(root, dir, "buggyline_location.json")
        buggy_lines_range = None
        with open(json_file, "r") as f:
            data = json.load(f)
            # extend at later point to also include transformed files (?)
            buggy_lines_range = range(data["original"][0][0], data["original"][0][1])
        with open(os.path.join(root, dir, dir + "_original_method.java"), "r") as f:
            line_count = 0
            for line in f:
                line_count += 1
                dataset.append(
                    Statement(
                        f.name,
                        line_count,
                        line.strip(),
                        line_count in buggy_lines_range,
                    )
                )
                max_length = max(max_length, len(dataset[-1].tokens[0]))

print(
    "Max length:", max_length
)  # 54 in this example, this seems long? maybe do not pad to max length?

for statement in dataset:
    statement.tokens = torch.nn.functional.pad(
        statement.tokens,  # tensor to be padded
        (
            0,
            max_length - statement.tokens.size(1),
        ),  # pad by how much on each side, 0 on left, max_length - statement.tokens.size(1) on right
        value=tokenizer.pad_token_id,  # this is the fill value for the padding
    )

print(dataset[0].tokens)

# For example get tokens of one statement
context_embeddings = model(dataset[0].tokens)[0]
# Determine the final size of the tensor
final_size = context_embeddings.size()
print("Final size:", final_size)
