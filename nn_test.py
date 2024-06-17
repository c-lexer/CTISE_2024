import torch
from transformers import RobertaModel,RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import numpy as np
from data_helpers import Datascraper
from plot_helpers import plot
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model.to(device)
datascraper = Datascraper(device)
datascraper.scrape_files(token_cutoff=4, nth_item=100)
# for data in datascraper.dataset:
#    data.pretty_print()
datascraper.pad_tokens(max_length=50)

features = []
labels = []
attention_masks = []
print("Finishing data preparation.")
for statement in datascraper.dataset:
    input_ids = statement.tokens.to(device)
    attention_mask = (input_ids != datascraper.tokenizer.pad_token_id).long().to(device)
    with torch.no_grad():
        # Extract contextual embeddings
        contextual_embeddings = model(input_ids, attention_mask=attention_mask)[0]
        # Mean pooling over the contextual embeddings and flatten
        pooled_features = (
            torch.mean(contextual_embeddings, dim=1).detach().cpu().numpy().flatten()
        )
        features.append(pooled_features)
        labels.append(statement.vulnerable)

print("Finishing embedding of tokens.")

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)
# Split the data into training and testing sets
# a split of 80/20 is recommended
features = features.reshape(features.shape[0], features.shape[1], -1)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
# Resample the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
X_train_resampled = X_train_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])

# USING NN
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Calculate class weights
class_counts = np.bincount(y_train_resampled)
class_weights = {i: len(y_train_resampled) / class_counts[i] for i in range(len(class_counts))}
pos_weight = torch.tensor(class_weights[1] / class_weights[0], dtype=torch.float32).to(device)

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define model, criterion, and optimizer

hidden_size = 128
num_layers = 2
output_size = 1
input_size = X_train_tensor.shape[2]
model_lstm = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
#model_nn = SimpleNN(input_size).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model_lstm.parameters(), lr=0.001)
# Training the neural network
num_epochs = 20
for epoch in range(num_epochs):
    for i, (features_batch, labels_batch) in enumerate(train_loader):
        outputs = model_lstm(features_batch).squeeze()
        loss = criterion(outputs, labels_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model_lstm.eval()
with torch.no_grad():
    y_train_pred = model_lstm(X_train_tensor).squeeze().cpu().numpy()
    y_test_pred = model_lstm(X_test_tensor).squeeze().cpu().numpy()

# Binarize the predictions
y_train_pred = (y_train_pred > 0.5).astype(int)
y_test_pred = (y_test_pred > 0.5).astype(int)

#Calculate and print metrics
accuracy = accuracy_score(y_test, y_test_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_test_pred))

# ROC and AUC
y_scores = model_lstm(X_test_tensor).squeeze().detach().cpu().numpy()
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plot(fpr, tpr, roc_auc)


