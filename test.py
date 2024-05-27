import torch
from transformers import RobertaModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import numpy as np
from data_helpers import Datascraper
from plot_helpers import plot
from scipy.stats import randint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)

datascraper = Datascraper()
datascraper.scrape_files(token_cutoff=4)
datascraper.pad_tokens(max_length=50)

features = []
labels = []

for statement in datascraper.dataset:
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
