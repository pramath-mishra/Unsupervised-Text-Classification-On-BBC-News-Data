import warnings
warnings.filterwarnings("ignore")

import spacy
import string
import pathlib
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn import metrics
from sklearn import neighbors
from tabulate import tabulate


class Classifier:

    def __init__(self, distance_metric="euclidean"):
        self.distance_metric = distance_metric
        self.nlp = spacy.load('en_core_web_lg')

    @staticmethod
    def clean_text(text):
        return ' '.join(
            text.lower().translate(
                str.maketrans('', '', string.punctuation)
            ).replace('\n', ' ').split()
        )

    def embed(self, tokens):
        lexemes = (self.nlp.vocab[token] for token in tokens)

        vectors = np.asarray([
            lexeme.vector
            for lexeme in lexemes
            if lexeme.has_vector and not lexeme.is_stop and len(lexeme.text) > 1
        ])

        if len(vectors) > 0:
            centroid = vectors.mean(axis=0)
        else:
            width = self.nlp.meta['vectors']['width']
            centroid = np.zeros(width)

        return centroid

    def label_neighbor_fit(self):
        label_vectors = np.asarray([self.embed(label_.split(' ')) for label_ in label_names])
        neigh = neighbors.NearestNeighbors(
            n_neighbors=1,
            metric=spatial.distance.cosine if self.distance_metric == "cosine" else spatial.distance.euclidean
        )
        neigh.fit(label_vectors)
        return neigh

    def predict(self, doc):
        neigh = self.label_neighbor_fit()
        doc = self.clean_text(doc)
        tokens = doc.split(' ')[:50]
        centroid = self.embed(tokens)
        closest_label = neigh.kneighbors([centroid], return_distance=False)[0][0]
        return closest_label


if __name__ == '__main__':

    # loading bbc datasets
    docs = list()
    labels = list()

    directory = pathlib.Path('bbc')
    label_names = ['business', 'entertainment', 'politics', 'sport', 'technology']

    for label in label_names:
        for file in directory.joinpath(label).iterdir():
            labels.append(label)
            docs.append(file.read_text(encoding='unicode_escape'))
    print("bbc news data loaded...\n -sample:", docs[0])

    # inference using cosine distance
    obj = Classifier(distance_metric="cosine")
    predictions = [label_names[obj.predict(doc)] for doc in docs]

    # classification report
    report = metrics.classification_report(y_true=labels, y_pred=predictions, labels=label_names, output_dict=True)
    report = pd.DataFrame([
        {
            "label": key,
            "precision": value["precision"],
            "recall": value["recall"],
            "support": value["support"]
        }
        for key, value in report.items()
        if key not in ["accuracy", "macro avg", "weighted avg"]
    ])
    print(
        tabulate(
            report,
            headers="keys",
            tablefmt="psql"
        ),
        file=open("./classification_report_cosine_distance.txt", "w")
    )

    # inference using euclidean distance
    obj = Classifier(distance_metric="euclidean")
    predictions = [label_names[obj.predict(doc)] for doc in docs]

    # classification report
    report = metrics.classification_report(y_true=labels, y_pred=predictions, labels=label_names, output_dict=True)
    report = pd.DataFrame([
        {
            "label": key,
            "precision": value["precision"],
            "recall": value["recall"],
            "support": value["support"]
        }
        for key, value in report.items()
        if key not in ["accuracy", "macro avg", "weighted avg"]
    ])
    print(
        tabulate(
            report,
            headers="keys",
            tablefmt="psql"
        ),
        file=open("./classification_report_euclidean_distance.txt", "w")
    )
