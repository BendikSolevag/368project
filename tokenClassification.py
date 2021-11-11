import Models
import numpy as np
import torch
from conllu import parse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, accuracy_score
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class FormatData(torch.utils.data.Dataset):

    def __init__(self, data, tag_col='upos'):
        self.formatted_data = [[(token['form'], str(token[tag_col]), token['misc']) for token in sentence] for sentence in data]

    def __getitem__(self, index):
        return self.formatted_data[index]

    def __len__(self):
        return len(self.formatted_data)


def get_data():
    with open("Data/pos_tagging/no_bokmaal-ud-train.conllu.txt", 'r', encoding="utf-8") as file:
        nb_train = file.read()
    with open("Data/pos_tagging/no_nynorsk-ud-train.conllu.txt", 'r', encoding="utf-8") as file:
        no_train = file.read()

    with open("Data/pos_tagging/no_bokmaal-ud-test.conllu.txt", 'r', encoding="utf-8") as file:
        nb_test = file.read()
    with open("Data/pos_tagging/no_nynorsk-ud-test.conllu.txt", 'r', encoding="utf-8") as file:
        no_test = file.read()

    sentences = parse(nb_test)[:10]
    test_len_nb = len(sentences)
    sentences_train = parse(nb_train)[:10]
    sentences.extend(sentences_train)

    sentences_no = parse(no_test)[:10]
    test_len_no = len(sentences_no)
    sentences_no_train = parse(no_train)[:10]
    sentences_no.extend(sentences_no_train)
    return sentences, test_len_nb, sentences_no, test_len_no


def tokenize_and_preserve_labels(sentence, tokenizer, tag=False):
    tokenized_sentence = []
    labels = []

    for word, w_tag, label in sentence:

        if tag:
            label = w_tag
        else:
            label = label['name']

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def get_tag(sentences, tag=False):
    tag_values = []
    for sentence in sentences:
        for s, t, l in sentence:
            if tag:
                tag_values.append(t)
            else:
                tag_values.append(l['name'])

    tag_values = list(set(tag_values))
    tag_values.append("PAD")
    return tag_values


def get_tag2idx(set_of_t):
    return {t: i for i, t in enumerate(set_of_t)}, set_of_t


def data_helper_tokenize_and_format(tokenizer, sentences, tag2idx, tag=False):
    MAX_LEN = 75

    tokenized_bert_text_and_labels = [tokenize_and_preserve_labels(sentence, tokenizer, tag=tag)
                                      for sentence in sentences]

    bert_tokenized_text = [t for t, l in tokenized_bert_text_and_labels]
    bert_labels = [l for t, l in tokenized_bert_text_and_labels]

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in bert_tokenized_text],
                              maxlen=MAX_LEN, dtype='long', value=0.0, truncating='post',
                              padding='post')
    tags = pad_sequences([[tag2idx.get(l) for l in label] for label in bert_labels],
                         maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                         dtype="long", truncating="post")
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    return input_ids, tags, attention_masks


def data_helper_torch_datasets(input_ids, tags, attention_masks, test_len):
    test_input, train_input = input_ids[:test_len], input_ids[test_len:]
    test_tags, train_tags = tags[:test_len], tags[test_len:]
    test_masks, train_masks = attention_masks[:test_len], attention_masks[test_len:]

    train_inputs = torch.tensor(train_input)
    test_inputs = torch.tensor(test_input)
    train_tags = torch.tensor(train_tags).long()
    test_tags = torch.tensor(test_tags).long()
    train_masks = torch.tensor(train_masks)
    test_masks = torch.tensor(test_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=8)

    test_data = TensorDataset(test_inputs, test_masks, test_tags)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=8)
    return train_dataloader, test_dataloader


def train_and_test_model_on_ner(pipeline, name, sentences, tag2idx, tag_values, test_len, epochs=3, tag=False):

    model = pipeline.model

    input_ids, tags, attention_masks = data_helper_tokenize_and_format(pipeline.tokenizer, sentences, tag2idx, tag=tag)
    train_dataloader, test_dataloader = data_helper_torch_datasets(input_ids, tags, attention_masks, test_len)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )
    max_grad_norm = 1.0

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_values, testing_loss_values = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    f1, acc = 0, 0
    for _ in trange(epochs, desc="Epoch"):

        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0]
            loss.backward()

            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        loss_values.append(avg_train_loss)

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        predictions, true_labels = [], []
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)

            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(test_dataloader)
        testing_loss_values.append(eval_loss)
        print(f"Model: {name} scores")
        print("testing loss: {}".format(eval_loss))
        pred_tags = []
        test_tags = []
        for p, l in zip(predictions, true_labels):
            curr_p = []
            curr_l = []
            for p_i, l_i in zip(p, l):
                if tag_values[l_i] != "PAD":
                    curr_p.append(tag_values[p_i])
                    curr_l.append(tag_values[l_i])
            pred_tags.append(curr_p)
            test_tags.append(curr_l)

        acc = accuracy_score(pred_tags, test_tags)
        f1 = f1_score(pred_tags, test_tags)
        print("testing Accuracy: {}".format(accuracy_score(pred_tags, test_tags)))
        print("testing F1-Score: {}".format(f1_score(pred_tags, test_tags)))
    return f1, acc


def console_in():
    return input("Do you wish do do [N]amed entity recognition, [P]art of speech tagging or [D]ependency parsing?")


def run():

    question = True
    tag = False
    tag_col = 'upos'
    while question:
        ans = console_in()
        if ans.lower() == 'n':
            tag = False
            break
        elif ans.lower() == 'p':
            tag = True
            tag_col = 'upos'
            break
        elif ans.lower() == 'd':
            tag = True
            tag_col = 'head'
            break
        print("Pleas press either [n], [p] or [d].")

    sentences, test_len, sentences_no, test_len_no = get_data()

    sentences = FormatData(sentences)
    sentences_no = FormatData(sentences_no)
    tag2idx, tag_values = get_tag2idx(get_tag(sentences, tag=tag))
    tag2idx_no, tag_values_no = get_tag2idx(get_tag(sentences_no, tag=tag))
    f1_scores, acc_scores = {}, {}
    print("Starting training and testing on bokmål")
    f1_scores['norbert_bm'], acc_scores['norbert_bm'] = train_and_test_model_on_ner(Models.get_nor_bert(len(tag2idx), task='ner'), "NOR-BERT bokmål", sentences, tag2idx, tag_values, test_len, tag=tag)
    f1_scores['nbbert_bm'], acc_scores['nbbert_bm'] = train_and_test_model_on_ner(Models.get_nb_bert(len(tag2idx), task='ner'), "NB-BERT bokmål", sentences, tag2idx, tag_values, test_len, tag=tag)
    f1_scores['mbert_bm'], acc_scores['mbert_bm'] = train_and_test_model_on_ner(Models.get_mbert(len(tag2idx), task='ner'), "mBert bokmål", sentences, tag2idx, tag_values, test_len, tag=tag)
    print("\nTraining and testing on nynorsk")
    f1_scores['norbert_nn'], acc_scores['norbert_nn'] = train_and_test_model_on_ner(Models.get_nor_bert(len(tag2idx), task='ner'), "NOR-BERT nynorsk", sentences_no, tag2idx_no, tag_values_no, test_len_no, tag=tag)
    f1_scores['nbbert_nn'], acc_scores['nbbert_nn'] = train_and_test_model_on_ner(Models.get_nb_bert(len(tag2idx), task='ner'), "NB-BERT nynorsk", sentences_no, tag2idx_no, tag_values_no, test_len_no, tag=tag)
    f1_scores['mbert_nn'],  acc_scores['mbert_nn'] = train_and_test_model_on_ner(Models.get_mbert(len(tag2idx), task='ner'), "mBert nynorsk", sentences_no, tag2idx_no, tag_values_no, test_len_no, tag=tag)