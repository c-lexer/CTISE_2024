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
    def __init__(self) -> None:
        self.dataset = []
        self.root_dir = "."
        self.max_length = 0
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        pass

    def scrape_files(self, token_cutoff=0) -> None:
        with open("data.json", "r") as f:
            data = json.load(f)
            for entry in data:
                if entry["vul"] == 1:
                    buggy_lines_range = entry["flaw_line_no"]
                else:
                    buggy_lines_range = []
                line_count = 0
                split_lines = (entry["code"].split("\n"))
                for line in split_lines:
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
