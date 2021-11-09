from simple_elmo import ElmoModel
from transformers import pipeline, AutoTokenizer, BertForTokenClassification


def get_nor_elmo(path="Vectors/218.zip"):
    nor_elmo = ElmoModel()
    nor_elmo.load(path)
    return nor_elmo


def get_nor_bert(task="sentiment-analysis"):
    tokenizer = AutoTokenizer.from_pretrained("ltgoslo/norbert")
    return pipeline(task, tokenizer=tokenizer, model='ltgoslo/norbert')


def get_nb_bert(task="sentiment-analysis"):
    tokenizer = AutoTokenizer.from_pretrained('NbAiLab/nb-bert-base-mnli')
    return pipeline(task, tokenizer=tokenizer, model='NbAiLab/nb-bert-base-mnli')


def get_mbert(num_labels, task="sentiment-analysis"):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)
    return pipeline(task, tokenizer=tokenizer, model=model)
