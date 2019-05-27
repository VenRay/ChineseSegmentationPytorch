import json

import re

from random import shuffle


def save(path, texts):
    with open(path, 'w') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)


def clean(text):
    text = re.sub('\d{8}-\d{2}-\d{3}-\d{3}', '', text)
    text = re.sub('\[', '', text)
    return re.sub('/\S+', '', text)


def prepare(path_univ, path_train, path_dev, path_test):
    texts = list()
    with open(path_univ, 'r') as f:
        for line in f:
            text = clean(line).strip()
            if text:
                text = re.sub('  ', ' ', text)
                texts.append(text)
    shuffle(texts)
    bound1 = int(len(texts) * 0.9)
    bound2 = int(len(texts) * 1)
    save(path_train, texts[:bound1])
    save(path_dev, texts[bound1:bound2])
    save(path_test, texts[bound2:])


if __name__ == '__main__':
    path_univ = 'data/AS.utf8'
    path_train = 'data/train.json'
    path_dev = 'data/dev.json'
    path_test = 'data/test.json'
    prepare(path_univ, path_train, path_dev, path_test)
#将txt 打乱后 train 70% / dev 20% / test 10% 划分