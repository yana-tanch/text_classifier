import re
import nltk
import torch


def preprocess_text(text, stopwords=None, flg_stemm=False, flg_lemm=True):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    lst_text = text.split()

    if stopwords is not None:
        lst_text = [word for word in lst_text if word not in stopwords]

    if flg_stemm:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    if flg_lemm:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    text = ' '.join(lst_text)

    return text


def collate_fn(batch, vocab, tokenizer):
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)] # text -> вектор частот слов

    label_list, batch_voc, offsets = [], [], [0]

    for (label, text) in batch:
        label_list.append(label)
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        batch_voc.append(processed_text)
        offsets.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    batch_voc = torch.cat(batch_voc)  # частоты слов в баче

    return label_list, batch_voc, offsets