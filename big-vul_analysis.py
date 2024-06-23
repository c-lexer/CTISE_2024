import json
import statistics
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
def analyze_code_snippet(snippet):
    lines = snippet.split('\n')
    total_lines = len(lines)
    non_empty_lines = [line for line in lines if line.strip() != '']
    #number of non-empty lines
    num_non_empty_lines = len(non_empty_lines)
    #number of empty lines
    num_empty_lines = total_lines - num_non_empty_lines
    has_empty_lines = num_empty_lines > 0

    #tokenize each line and count the tokens
    token_counts = [len(tokenizer.tokenize(line)) for line in non_empty_lines]

    return {
        "total_lines": total_lines,
        "non_empty_lines": num_non_empty_lines,
        "empty_lines": num_empty_lines,
        "has_empty_lines": has_empty_lines,
        "token_counts": token_counts
    }

with open("data.json", "r") as f:
    data = json.load(f)

#analyze each code snippet
analysis_results = [analyze_code_snippet(entry["code"]) for entry in data]

#mean summary
mean_num_lines = statistics.mean([result['total_lines'] for result in analysis_results])
mean_num_tokens_per_line = statistics.mean([token for result in analysis_results for token in result['token_counts']])

# snippets with and without empty lines
snippets_with_empty_lines = sum(1 for result in analysis_results if result['has_empty_lines'])
snippets_without_empty_lines = len(analysis_results) - snippets_with_empty_lines


for i, result in enumerate(analysis_results[:5]):  # Print first 5 for brevity
    print(f"Snippet {i + 1} Analysis:")
    print(f"Total lines: {result['total_lines']}")
    print(f"Non-empty lines: {result['non_empty_lines']}")
    print(f"Empty lines: {result['empty_lines']}")
    print(f"Contains empty lines: {result['has_empty_lines']}")
    print(f"Token counts per line: {result['token_counts']}")
    print()


print(f"Mean number of lines per snippet: {mean_num_lines:.2f}")
print(f"Mean number of tokens per non-empty line: {mean_num_tokens_per_line:.2f}")

#count of snippets with and without empty lines
print(f"Number of snippets with empty lines: {snippets_with_empty_lines}")
print(f"Number of snippets without empty lines: {snippets_without_empty_lines}")