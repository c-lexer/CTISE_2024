from transformers import RobertaTokenizer
import os
import json
import torch


class Statement:
    def __init__(self, origin, line_nr, code_statement, vulnerable, tokens) -> None:
        self.origin = origin
        self.line_nr = line_nr
        self.code_statement = code_statement
        self.vulnerable = vulnerable
        self.tokens = tokens

    def pretty_print(self):
        print(
            f"Origin: {self.origin}\nLine number: {self.line_nr}\nCode statement: {self.code_statement}\nVulnerable: {self.vulnerable}\nTokens: {self.tokens}\n\n"
        )


class Datascraper:
    def __init__(self, root_dir="./llm-vul/VJBench-trans") -> None:
        self.dataset = []
        self.root_dir = root_dir
        self.max_length = 0
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        pass

    def scrape_files(self, token_cutoff=0) -> None:
        for root, dirs, files in os.walk(self.root_dir):
            for dir in dirs:
                json_file = os.path.join(root, dir, "buggyline_location.json")
                buggy_lines_range = None
                suffixes = [
                    ("original", "_original_method.java"),
                    ("structure_change_only", "_code_structure_change_only.java"),
                    ("rename+code_structure", "_full_transformation.java"),
                    ("rename_only", "_rename_only.java"),
                ]
                for bug_line, suffix in suffixes:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        # extend at later point to also include transformed files (?)
                        buggy_lines_range = range(
                            data[bug_line][0][0], data[bug_line][0][1]
                        )
                    with open(os.path.join(root, dir, dir + suffix), "r") as f:
                        line_count = 0
                        for line in f:
                            line_count += 1
                            tokens = self.tokenizer.encode(
                                line.strip(), return_tensors="pt"
                            )
                            if (
                                len(tokens[0]) > token_cutoff
                            ):  # ignore lines with less than 4 tokens
                                self.dataset.append(
                                    Statement(
                                        f.name,
                                        line_count,
                                        line.strip(),
                                        line_count in buggy_lines_range,
                                        self.tokenizer.encode(
                                            line.strip(), return_tensors="pt"
                                        ),
                                    )
                                )
                                self.max_length = max(
                                    self.max_length, len(self.dataset[-1].tokens[0])
                                )

    def pad_tokens(self, max_length=None):
        max_length = max_length if max_length is not None else self.max_length
        for statement in self.dataset:
            if statement.tokens.size(1) > max_length:
                statement.tokens = statement.tokens[:, :max_length]
            else:
                statement.tokens = torch.nn.functional.pad(
                    statement.tokens,  # tensor to be padded
                    (
                        0,
                        max_length - statement.tokens.size(1),
                    ),  # pad by how much on each side, 0 on left, max_length - statement.tokens.size(1) on right
                    value=self.tokenizer.pad_token_id,  # this is the fill value for the padding
                )
