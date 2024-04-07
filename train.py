import json
import argparse
from random import shuffle
from collections import Counter

import numpy as np
from sklearn import metrics

from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

# pip install -U spacy
# python -m spacy download ru_core_news_sm

from model import TextClassificationModel
from utils import collate_fn


def load_corpus(file, max_no_program=200):
    program_corpus = []
    no_program_corpus = []

    with open(file) as fp:
        data = json.load(fp)

        for d in data:
            if d['classname'] == 'no_program':
                no_program_corpus.append((0, d['text']))
            elif d['classname'] == 'program':
                program_corpus.append((1, d['text']))
            else:
                continue

    shuffle(no_program_corpus) #  рандомно
    size = min(max_no_program, len(no_program_corpus))
    no_program_corpus = no_program_corpus[:size]

    corpus = no_program_corpus + program_corpus

    return corpus


def calc_metrics(predictions, targets):
    confusion_matrix = metrics.confusion_matrix(targets, predictions)

    tp = confusion_matrix[1, 1]

    fp = confusion_matrix[1, 0]
    fn = confusion_matrix[0, 1]

    eps = 1e-6

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    print("p")
    print(precision)
    print("r")
    print(recall)



    f1_score = 2 * precision * recall / (precision + recall + eps)

    return f1_score, confusion_matrix

# pytorch.org
def main(args):
    corpus = load_corpus(args.dataset, max_no_program=args.max_no_program)

    num_train = int(len(corpus) * 0.7)
    train_dataset, valid_dataset = random_split(corpus, [num_train, len(corpus) - num_train])

    counter = Counter()
    tokenizer = get_tokenizer('spacy', 'ru_core_news_sm')

    for (_, batch_voc) in corpus:
        counter.update(tokenizer(batch_voc)) # отдельные слова

    vocab = Vocab(counter)


    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, vocab, tokenizer))
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, vocab, tokenizer))

    vocab_size = len(vocab)


    net = TextClassificationModel(vocab_size, args.embed_dim)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1) # stochastic gradient descend метод обратного распространения ошибок

    f1_best = 0
    conf_mat_best = None

    for epoch in range(args.num_epochs):
        # training
        net.train() # позволяет оптимизировать параметры сетки
        train_loss = 0 # средний в этой эпохе

        # train_dataset (list of tuples (label, text)) -> random batch (batch_size) -> collate_fn(batch, vocab, tokenizer) -> label, text, offsets
        for (label, batch_voc, offsets) in train_loader:

            optimizer.zero_grad() # обнуление градиентов параметров сети

            pred = net.forward(batch_voc, offsets) # вызывается ф-я forward

            loss = loss_fn(pred, label)

            # обновляем параметры сети с помощью алгоритма обратного распространения ошибок (backpropogation)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader) # делим на число бачей в train dataset

        # validation
        net.eval()  # фиксирует параметры сетки
        valid_loss = 0

        targets = np.empty((0,))
        predictions = np.empty((0,))

        # valid_dataset (list of tuples (label, text)) -> random batch (batch_size) -> collate_fn(batch, vocab, tokenizer) -> label, text, offsets
        for (label, batch_voc, offsets) in valid_loader:

            pred = net.forward(batch_voc, offsets)  # вызывается ф-я forward
            loss = loss_fn(pred, label)

            valid_loss += loss.item()

            target = label.numpy()
            prediction = pred.argmax(1).numpy()

            targets = np.concatenate((targets, target))
            predictions = np.concatenate((predictions, prediction))

        valid_loss /= len(valid_loader) # делим на число бачей в valid dataset

        f1_score, confusion_matrix = calc_metrics(predictions, targets)

        if f1_best < f1_score:
            f1_best = f1_score
            conf_mat_best = confusion_matrix


            state = dict(model_state=net.state_dict(),
                         vocab_size=vocab_size,
                         embed_dim=args.embed_dim,
                         vocab=vocab)

            torch.save(state, args.checkpoint)

        print(f'\nepoch: {epoch} train_loss: {train_loss:.4f} valid_loss: {valid_loss:.4f} valid_f1: {f1_best:.4f}')
        print(conf_mat_best)
        print("f")
        print(f1_score)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')

    parser.add_argument('--dataset', type=str, default='data/dataset.json')
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--embed-dim', type=int, default=600)
    parser.add_argument('--max-no-program', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, default='data/model.pth')

    args = parser.parse_args()

    main(args)
