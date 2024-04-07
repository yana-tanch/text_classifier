import json
import argparse
from pathlib import Path

import nltk

from utils import preprocess_text


def main(args):
    nltk.download('stopwords')
    nltk.download('wordnet')

    stopwords = nltk.corpus.stopwords.words('russian')

    dataset = []

    for file in Path(args.work_programs).glob('**/*.json'):
        print(f'process file: {file}')

        with open(file) as fp:
            program = json.load(fp)
            text = ''

            for key, value in program.items():
                text += value['title']
                text += value['text']

            text = preprocess_text(text, stopwords)
            dataset.append(dict(classname='program', text=text))

    print(f'process file: {args.common_texts}')

    with open(args.common_texts) as fp:
        data = json.load(fp)

        for d in data:
            text = preprocess_text(d['text'], stopwords)
            dataset.append(dict(classname='no_program', text=text))

    with open(args.dataset, 'w') as fp:
        json.dump(dataset, fp, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Dataset')

    parser.add_argument('--work-programs', type=str, default='data/work_programs')
    parser.add_argument('--common-texts', type=str, default='data/ru_sentiment.json')
    parser.add_argument('--dataset', type=str, default='data/dataset.json')

    args = parser.parse_args()

    main(args)

