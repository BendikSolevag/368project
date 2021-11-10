from simple_elmo import ElmoModel
from transformers import pipeline, AutoTokenizer, BertForTokenClassification


def get_nor_elmo(path="Vectors/218.zip"):
    nor_elmo = ElmoModel()
    nor_elmo.load(path)
    return nor_elmo


def get_nor_bert(num_labels, task="sentiment-analysis", model_type=BertForTokenClassification):
    tokenizer = AutoTokenizer.from_pretrained("ltgoslo/norbert")
    model = model_type.from_pretrained("ltgoslo/norbert", num_labels=num_labels)
    return pipeline(task, tokenizer=tokenizer, model=model)


def get_nb_bert(num_labels, task="sentiment-analysis", model_type=BertForTokenClassification):
    tokenizer = AutoTokenizer.from_pretrained('NbAiLab/nb-bert-large')
    model = model_type.from_pretrained('NbAiLab/nb-bert-large', num_labels=num_labels)
    return pipeline(task, tokenizer=tokenizer, model=model)


def get_mbert(num_labels, task="sentiment-analysis", model_type=BertForTokenClassification):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = model_type.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)
    return pipeline(task, tokenizer=tokenizer, model=model)
