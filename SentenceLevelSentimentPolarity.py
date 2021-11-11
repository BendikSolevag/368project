""" Fetch Models """
import torch
import Models
from transformers import AdamW, AutoTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def number_sentiment(sentiment):
    """
    Returns a number representation of the input.
        Parameters:
            sentiment (string): One of 'Positive' or 'Negative'

        Returns:
            1 if sentiment == 'Positive', 0 otherwise.
    """
    if sentiment == 'Positive':
        return 1
    return 0

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

def fetch_datasets():
    """
    Fetches data from the /Data directory. Parses labels, tokenizes inputs. Loads data into a custom pytorch Dataset 
        Returns:
            Six SentinentPolarityDataset datasets.
    """

    with open('./Data/sentence_level_sentiment_polarity/train.json') as polarity_data:
        polarity_array = json.load(polarity_data)[:5]
        train_texts = [datapoint['text'] for datapoint in polarity_array]
        train_labels = [number_sentiment(datapoint['label']) for datapoint in polarity_array]
        
    with open('./Data/sentence_level_sentiment_polarity/val.json') as polarity_data:
        polarity_array = json.load(polarity_data)[:5]
        val_texts = [datapoint['text'] for datapoint in polarity_array]
        val_labels = [number_sentiment(datapoint['label']) for datapoint in polarity_array]
        
    with open('./Data/sentence_level_sentiment_polarity/test.json') as polarity_data:
        polarity_array = json.load(polarity_data)[:5]
        test_texts = [datapoint['text'] for datapoint in polarity_array]
        test_labels = [number_sentiment(datapoint['label']) for datapoint in polarity_array]


    """ Parse data into datasets """
    nor_bert_tokenizer = AutoTokenizer.from_pretrained("ltgoslo/norbert")
    nb_bert_tokenizer = AutoTokenizer.from_pretrained('NbAiLab/nb-bert-base')
    mbert_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')


    nor_bert_train_encodings = nor_bert_tokenizer(train_texts, truncation=True, padding=True)
    nb_bert_train_encodings = nb_bert_tokenizer(train_texts, truncation=True, padding=True)
    mbert_train_encodings = mbert_tokenizer(train_texts, truncation=True, padding=True)

    nor_bert_val_encodings = nor_bert_tokenizer(val_texts, truncation=True, padding=True)
    nb_bert_val_encodings = nb_bert_tokenizer(val_texts, truncation=True, padding=True)
    mbert_val_encodings = mbert_tokenizer(val_texts, truncation=True, padding=True)

    nor_bert_test_encodings = nor_bert_tokenizer(test_texts, truncation=True, padding=True)
    nb_bert_test_encodings = nb_bert_tokenizer(test_texts, truncation=True, padding=True)
    mbert_test_encodings = mbert_tokenizer(test_texts, truncation=True, padding=True)


    nor_bert_train_dataset = SentinentPolarityDataset(nor_bert_train_encodings, train_labels)
    nb_bert_train_dataset = SentinentPolarityDataset(nb_bert_train_encodings, train_labels)
    mbert_train_dataset = SentinentPolarityDataset(mbert_train_encodings, train_labels)

    nor_bert_test_dataset = SentinentPolarityDataset(nor_bert_test_encodings, test_labels)
    nb_bert_test_dataset = SentinentPolarityDataset(nb_bert_test_encodings, test_labels)
    mbert_test_dataset = SentinentPolarityDataset(mbert_test_encodings, test_labels)
    return nor_bert_train_dataset, nb_bert_train_dataset, mbert_train_dataset, nor_bert_test_dataset, nb_bert_test_dataset, mbert_test_dataset

""" Tune models """



def tune(model, optim, dataset):
    """
    Trains a given model on the given dataset using the given optimizer. 
        Parameters:
            model (pytorch model): The model we want to train.
            optim (pytoch optimizer): The optimizer we wish to use.
            dataset (pytorch dataset): The dataset we wish to tune our model to.
    """
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model.train()
    for epoch in range(3):
        for batch in tqdm(loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
    model.eval()

def f1_score(TP, FP, FN):
    """
    Calculates F1 score.
        Parameters:
            TP (int): Amount of true positives
            FP (int): Amount of false positives
            FN (int): Amount of false negatives
        Returns:
            Calculated F1 score (float)
    """
    return TP / (TP + (0.5 * (FP + FN)))

def accuracy(TP, TN, FP, FN):
    """
    Calculates accuracy
        Parameters:
            TP (int): True positives
            TN (int): True negatives
            FP (int): False positives
            FN (int): False negatives
        Returns:
            Calculated accuracy (float)
    """
    return (TP + TN) / (TP + TN + FP + FN)
    
def eval(model, dataset):
    """
    Evaluates given model on given dataset.
        Params:
            model (pytorch model): The model we wish to evaluate.
            dataset (pytorch dataset): The dataset used to evaluate the model.
        Returns:
            F1 score and accuracy of the given model on the given dataset.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        if(outputs['logits'][0][0] > outputs['logits'][0][1]):
            # False prediction
            if labels[0] == 1:
                FN += 1
            else:
                TN += 1
        else:
            # True prediction
            if labels[0] == 1:
                TP += 1
            else:
                FP += 1
    return f1_score(TP, FP, FN), accuracy(TP, TN, FP, FN)

def run():
    nb_bert_pipe = Models.get_nb_bert(2, model_type=BertForSequenceClassification)
    mbert_pipe = Models.get_mbert(2, model_type=BertForSequenceClassification)
    nor_bert_pipe = Models.get_nor_bert(2, model_type=BertForSequenceClassification)

    nor_bert_train_dataset, nb_bert_train_dataset, mbert_train_dataset, nor_bert_test_dataset, nb_bert_test_dataset, mbert_test_dataset = fetch_datasets()

    nor_bert_model=nor_bert_pipe.model
    nor_bert_model.to(device)
    nor_bert_optim = AdamW(nor_bert_model.parameters(), lr=5e-5)
    tune(nor_bert_model, nor_bert_optim, nor_bert_train_dataset)
    nor_bert_f1, nor_bert_accuracy = eval(nor_bert_model, nor_bert_test_dataset)

    nb_bert_model = nb_bert_pipe.model
    nb_bert_model.to(device)
    nb_bert_optim = AdamW(nb_bert_model.parameters(), lr=5e-5)
    tune(nb_bert_model, nb_bert_optim, nb_bert_train_dataset)
    nb_bert_f1, nb_bert_accuracy = eval(nb_bert_model, nb_bert_test_dataset)


    mbert_model = mbert_pipe.model
    mbert_model.to(device)
    mbert_optim = AdamW(mbert_model.parameters(), lr=5e-5)
    tune(mbert_model, mbert_optim, mbert_train_dataset)
    mbert_f1, mbert_accuracy = eval(mbert_model, mbert_test_dataset)

    with open('./results/sentence_level_sentiment_polarity.txt', 'w') as file:
        file.write('NorBert - F1 score: ' + str(nor_bert_f1) + ' Accuracy: ' + str(nor_bert_accuracy) + '\n')
        file.write('NbBert - F1 score: ' + str(nb_bert_f1) + ' Accuracy: ' + str(nb_bert_accuracy) + '\n')
        file.write('mBert - F1 score: ' + str(mbert_f1) + ' Accuracy: ' + str(mbert_accuracy) + '\n')