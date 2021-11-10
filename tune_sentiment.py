import torch
import Models
import json
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer


nb_bert_pipe = Models.get_nb_bert()
mbert_pipe = Models.get_mbert()
nor_bert_pipe = Models.get_nor_bert()


def number_sentiment(sentiment):
    if sentiment == 'Positive':
        return 1
    return 0

with open('./norbench/polarity.json') as polarity_data:
    polarity_array = json.load(polarity_data)
    inputs = [datapoint['text'] for datapoint in polarity_array]
    labels = [number_sentiment(datapoint['label']) for datapoint in polarity_array]


train_texts, val_test_texts, train_labels, val_test_labels = train_test_split(inputs, labels, test_size=.2)
val_texts, test_texts, val_labels, test_labels = train_test_split(val_test_texts, val_test_labels, test_size=.5)

tokenizer = AutoTokenizer.from_pretrained('ltgoslo/norbert')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


class SentinentPolarityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SentinentPolarityDataset(train_encodings, train_labels)
val_dataset = SentinentPolarityDataset(val_encodings, val_labels)
test_dataset = SentinentPolarityDataset(test_encodings, test_labels)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model=nor_bert_pipe.model
model.to(device)
model.train()
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optim = AdamW(model.parameters(), lr=5e-5)
for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()

print(nb_bert_pipe('Dette er en positiv setning'))