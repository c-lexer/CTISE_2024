import os
import json
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

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

# print(
#    "Max length:", max_length
# )  # 54 in this example, this seems long? maybe do not pad to max length?

for statement in dataset:
    statement.tokens = torch.nn.functional.pad(
        statement.tokens,  # tensor to be padded
        (
            0,
            max_length - statement.tokens.size(1),
        ),  # pad by how much on each side, 0 on left, max_length - statement.tokens.size(1) on right
        value=tokenizer.pad_token_id,  # this is the fill value for the padding
    )

# print(dataset[0].tokens)

# For example get tokens of one statement
# context_embeddings = model(dataset[0].tokens)[0]
# Determine the final size of the tensor
# print(context_embeddings)
# final_size = context_embeddings.size()

# outputs Final size: torch.Size([1, 54, 768])
# which means 1 statement is processed, consisting of 54 tokens, each token has a vector of length 768 with its embeddings
# print("Final size:", final_size)


# Extract features and labels from the dataset
# split data into features and target, features meaning tokens and target meaning "the thing we try to classify"
features = []
labels = []

for statement in dataset:
    # Extract contextual embeddings
    contextual_embeddings = model(statement.tokens)[0]
    # Mean pooling over the contextual embeddings and flatten
    # ChatGPT helped me with this, apparently I need a 2 dimensional vector for a random forest classifier
    pooled_features = (
        torch.mean(contextual_embeddings, dim=1).detach().cpu().numpy().flatten()
    )
    features.append(pooled_features)
    labels.append(statement.vulnerable)


# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets
# a split of 80/20 is recommended
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.1, random_state=42
)

# Initialize the Random Forest classifier & train it
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
predictions = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, predictions))
# prints something like the following:
# Accuracy: 0.9571428571428572
# Classification Report:
#               precision    recall  f1-score   support
#
#        False       0.96      1.00      0.98       269
#         True       0.00      0.00      0.00        11
#
#     accuracy                           0.96       280
#    macro avg       0.48      0.50      0.49       280
# weighted avg       0.92      0.96      0.94       280
