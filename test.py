import json
import argparse

import nltk
import torch
from torchtext.data.utils import get_tokenizer

from utils import preprocess_text, collate_fn
from model import TextClassificationModel


def main(args):
    nltk.download('stopwords')
    nltk.download('wordnet')

    stopwords = nltk.corpus.stopwords.words('russian')

    state = torch.load(args.checkpoint)
    vocab = state['vocab']

    net = TextClassificationModel(state['vocab_size'], state['embed_dim'])
    net.load_state_dict(state['model_state'])  # параметры нейронной сети

    with open(args.test_file) as fp:
        program = json.load(fp)
        text = ''

        for key, value in program.items():
            text += value['title']
            text += value['text']

        text = preprocess_text(text, stopwords)

    tokenizer = get_tokenizer('spacy', 'ru_core_news_sm')

    batch = [(0, text)]
    _, batch_voc, offsets = collate_fn(batch, vocab,tokenizer)  # частоты слов и разделения

    net.eval()

    pred = net(batch_voc, offsets)

    if pred[0][1] > pred[0][0]:
        print(f'This is a work program.')
    else:
        print(f'This is not a work program.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Model')

    parser.add_argument('--checkpoint', type=str, default='data/model.pth')
    parser.add_argument('--test-file', type=str, default='data/work_programs/038689_Испанский язык_content.json')

    args = parser.parse_args()

    main(args)