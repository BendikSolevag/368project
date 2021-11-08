from simple_elmo import ElmoModel
from transformers import pipeline


def get_nor_elmo(path="Vectors/218.zip"):
    nor_elmo = ElmoModel()
    nor_elmo.load(path)
    return nor_elmo


def get_nor_bert(tokenizer="sentiment-analysis"):
    return pipeline(tokenizer, model='ltgoslo/norbert')


def get_nb_bert(tokenizer="sentiment-analysis"):
    return pipeline(tokenizer, model='NbAiLab/nb-bert-base-mnli')


def get_mbert(tokenizer="sentiment-analysis"):
    return pipeline(tokenizer, model='bert-base-multilingual-cased')
