import os
import json
import pandas as pd
from transformers import RobertaTokenizer
import matplotlib.pyplot as plt
import seaborn as sns


class Statement_llm:
    def __init__(self, file_name, line_number, code_line, vulnerable, tokens):
        self.file_name = file_name
        self.line_number = line_number
        self.code_line = code_line
        self.vulnerable = vulnerable
        self.tokens = tokens


class Datascraper_llm:
    def __init__(self, root_dir, device):
        self.root_dir = root_dir
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.dataset = []
        self.max_length = 0
        self.empty_lines_count = 0

        # Initialize min and max token length trackers
        self.min_token_length_vulnerable = float('inf')
        self.max_token_length_vulnerable = 0
        self.min_token_length_non_vulnerable = float('inf')
        self.max_token_length_non_vulnerable = 0

    def is_empty_line(self, line):
        # Check if line is empty or contains only whitespace
        return line.strip() == ''

    def scrape_files(self, token_cutoff=0) -> None:
        function_count = 0
        vulnerable_count = 0
        non_vulnerable_count = 0

        for root, dirs, files in os.walk(self.root_dir):
            for dir in dirs:
                json_file = os.path.join(root, dir, "buggyline_location.json")
                if not os.path.isfile(json_file):
                    print(f"JSON file {json_file} does not exist.")
                    continue

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
                        if bug_line not in data:
                            print(f"{bug_line} not found in {json_file}.")
                            continue
                        buggy_lines_range = range(
                            data[bug_line][0][0], data[bug_line][0][1]
                        )
                    java_file = os.path.join(root, dir, dir + suffix)
                    if not os.path.isfile(java_file):
                        print(f"Java file {java_file} does not exist.")
                        continue

                    with open(java_file, "r") as f:
                        line_count = 0
                        for line in f:
                            line_count += 1
                            tokens = self.tokenizer.encode(
                                line.strip(), return_tensors="pt"
                            )
                            if len(tokens[0]) > token_cutoff:
                                statement = Statement_llm(
                                    f.name,
                                    line_count,
                                    line.strip(),
                                    line_count in buggy_lines_range,
                                    tokens,
                                )
                                self.dataset.append(statement)
                                self.max_length = max(self.max_length, len(statement.tokens[0]))

                                # Check for empty lines
                                if self.is_empty_line(line.strip()):
                                    self.empty_lines_count += 1

                                # Update min and max token lengths
                                if statement.vulnerable:
                                    vulnerable_count += 1
                                    self.min_token_length_vulnerable = min(self.min_token_length_vulnerable, len(statement.tokens[0]))
                                    self.max_token_length_vulnerable = max(self.max_token_length_vulnerable, len(statement.tokens[0]))
                                else:
                                    non_vulnerable_count += 1
                                    self.min_token_length_non_vulnerable = min(self.min_token_length_non_vulnerable, len(statement.tokens[0]))
                                    self.max_token_length_non_vulnerable = max(self.max_token_length_non_vulnerable, len(statement.tokens[0]))

                function_count += 1

        print(f"Total functions: {function_count}")
        print(f"Total vulnerable lines: {vulnerable_count}")
        print(f"Total non-vulnerable lines: {non_vulnerable_count}")
        print(f"Min token length for vulnerable lines: {self.min_token_length_vulnerable}")
        print(f"Max token length for vulnerable lines: {self.max_token_length_vulnerable}")
        print(f"Min token length for non-vulnerable lines: {self.min_token_length_non_vulnerable}")
        print(f"Max token length for non-vulnerable lines: {self.max_token_length_non_vulnerable}")
        print(f"Total empty lines: {self.empty_lines_count}")

datascraper = Datascraper_llm(root_dir="/Users/atagilova/PycharmProjects/CTISE_2024/llm-vul/VJBench-trans", device="cpu")
datascraper.scrape_files(token_cutoff=4)
data = {
    "file_name": [s.file_name for s in datascraper.dataset],
    "line_number": [s.line_number for s in datascraper.dataset],
    "code_line": [s.code_line for s in datascraper.dataset],
    "vulnerable": [s.vulnerable for s in datascraper.dataset],
    "token_length": [len(s.tokens[0]) for s in datascraper.dataset]
}

df = pd.DataFrame(data)
#print(df.head())
mean_token_length_vulnerable = df[df["vulnerable"] == True]["token_length"].mean()
mean_token_length_non_vulnerable = df[df["vulnerable"] == False]["token_length"].mean()
print(f"Mean token length for vulnerable lines: {mean_token_length_vulnerable}")
print(f"Mean token length for non-vulnerable lines: {mean_token_length_non_vulnerable}")

sns.histplot(df, x="token_length", hue="vulnerable", multiple="stack")
plt.title("Token Length Distribution")
plt.show()
