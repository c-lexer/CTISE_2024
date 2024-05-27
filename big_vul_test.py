import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch

from tabulate import tabulate
data = pd.read_csv('MSR_data_cleaned.csv', nrows=4)
#print(tabulate(data.iloc[[0]], headers='keys', tablefmt='pretty'))
#print(data.at[1, 'func_before'] )
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel


# Initialize the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')


# Function to tokenize and encode each statement in the code
def tokenize_and_encode(code):
    # Tokenize the code
    tokens = tokenizer(code, return_tensors='pt', truncation=True, padding='max_length', max_length=50)
    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model(**tokens)
    # Get the last hidden state (embeddings)
    embeddings = outputs.last_hidden_state
    return embeddings


# Initialize lists to hold features and labels
features = []
labels = []

# Process each function in the DataFrame
for index, row in data.iterrows():
    code = row['func_before']
    label = row['vul']

    # Tokenize and encode the function code
    embeddings = tokenize_and_encode(code)

    # Mean pooling over the contextual embeddings and flatten
    pooled_features = torch.mean(embeddings, dim=1).detach().cpu().numpy().flatten()

    # Append to the lists
    features.append(pooled_features)
    labels.append(label)

# Display the features and labels
print("Features shape:", [f.shape for f in features])
print("Labels:", labels)
