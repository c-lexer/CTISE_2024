import pandas as pd
import json
from transformers import BertTokenizer, BertModel
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import numpy as np
from plot_helpers import plot
from scipy.stats import randint

#data = pd.read_csv('MSR_data_cleaned.csv', nrows=4)
#print(tabulate(data.iloc[[0]], headers='keys', tablefmt='pretty'))
# #print(data.at[1, 'func_before'] )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertModel.from_pretrained("microsoft/codebert-base")
model.to(device)


class CodeDataset:
    def __init__(self, json_path, max_length=50, num_items=4):
        self.json_path = json_path
        self.max_length = max_length
        self.num_items = num_items
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.dataset = self.load_data()

    def load_data(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        data = data[:self.num_items]
        return [{'code': item['code'], 'tokens': self.tokenize_code(item['code'])} for item in data]

    def tokenize_code(self, code):
        inputs = self.tokenizer(code, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def pad_tokens(self, max_length=None):
        max_length = max_length if max_length is not None else self.max_length
        for statement in self.dataset:
            if statement['tokens'].size(1) > max_length:
                statement['tokens'] = statement['tokens'][:, :max_length]
            else:
                statement['tokens'] = torch.nn.functional.pad(
                    statement['tokens'],  # tensor to be padded
                    (0, max_length - statement['tokens'].size(1)),  # pad by how much on each side
                    value=self.tokenizer.pad_token_id  # this is the fill value for the padding
                )

    def get_tensors(self):
        return [statement['tokens'] for statement in self.dataset]

# Initialize the dataset
code_dataset = CodeDataset(json_path='data.json', max_length=50, num_items=4)

# Pad the tokens in the dataset
code_dataset.pad_tokens()

# Get the list of tensors with the required shape
tensors = code_dataset.get_tensors()

# Verify the shape of the tensors
for i, tensor in enumerate(tensors):
    print(f"Tensor {i} shape: {tensor.shape}")
features = []
labels = []

for statement in CodeDataset.dataset:
    # Extract contextual embeddings
    contextual_embeddings = model(statement.tokens)[0]
    # Mean pooling over the contextual embeddings and flatten
    # ChatGPT helped me with this, apparently I need a 2 dimensional vector for a random forest classifier
    pooled_features = (
        torch.mean(contextual_embeddings, dim=2).detach().cpu().numpy().flatten()
    )
    features.append(pooled_features)
    labels.append(statement.vulnerable)


# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)
print (features[:3])
print(labels[:3])
# Split the data into training and testing sets
# a split of 80/20 is recommended
"""
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


# Initialize the Random Forest classifier & train it
param_dist = {"n_estimators": randint(50, 500), "max_depth": randint(1, 20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
rand_search.fit(X_train, y_train)

# Make predictions on the test set
predictions = rand_search.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
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


# Get the predicted probabilities for the positive class
y_scores = rand_search.predict_proba(X_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plot(fpr, tpr, roc_auc)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print("Best hyperparameters:", rand_search.best_params_)
"""