import numpy as np
import spacy

from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# This file contains functions that perform feature extraction for the word-distribution method


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.add_pipe("sentencizer")


def parse_book(path):
    """
    Reads file line by line and saves each line as a list of words in a set
    :param path: str
    :return: set <list<str>>
    """
    texts = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # Remove leading and trailing whitespaces
            line = " ".join(line.split())  # Remove extra spaces between words
            texts.append(line)
    texts_set = set(texts)
    return texts_set


def lemmatise_pipe(doc):
    """
    Performs lemmatisation on a document
    :param doc: generator
    :return: list <str>
    """
    lemmas = [str(token.lemma_).lower() for token in doc
              if token.is_alpha and not token.is_punct]
    return lemmas


def preprocess_pipe(texts):
    """
    Executes lemmatisation in parallel by calling nlp.pipe() and working on batches of documents
    :param texts:
    :return: list <list<str>>
    """
    lemmas_list = []
    for doc in nlp.pipe(texts, batch_size=20):
        lemmas_list.append(lemmatise_pipe(doc))
    return lemmas_list


def collect_frequency(lemma_list):
    """
    Counts occurrences of each word given a list and ranks them in popularity
    :param lemma_list: list <list<str>>
    :return: float, float
    """
    final_counter = Counter()
    for lemmas in lemma_list:
        counts = Counter(lemmas)
        final_counter.update(counts)  # Collect word occurrences
    final_counter = final_counter.most_common()  # Rank the words by popularity

    rank, freq = [], []
    for i, (_, f) in enumerate(final_counter, start=1):
        rank.append(i)
        freq.append(f)

    # Return the log 10 of the ranks and frequency for linear regression
    rank = np.log10(rank).reshape(-1, 1)
    freq = np.log10(freq)

    return rank, freq


def linear_regression(rank, freq_actual):
    """
    Performs linear regression given the x and y co-ordinates (rank and freq_actual)
    to find the gradient, coefficient of determination, and mean squared error
    :param rank: list <float>
    :param freq_actual: list <float>
    :return: float, float, float
    """
    linear_reg = LinearRegression()
    linear_reg.fit(rank, freq_actual)
    freq_pred = linear_reg.predict(rank)  # Predict frequencies of each rank according to the regression line

    slope = linear_reg.coef_.item()
    r2 = r2_score(freq_actual, freq_pred)
    mse = mean_squared_error(freq_actual, freq_pred)
    return [slope, r2, mse]


def word_distribution_feature_extraction(path):
    """
    Extracts the gradient, coefficient of determination, and mean squared error of a text
    :param path: str
    :return: float, float, float
    """
    texts_set = parse_book(path)
    lemmas_list = preprocess_pipe(texts_set)
    rank, freq_actual = collect_frequency(lemmas_list)
    features = linear_regression(rank, freq_actual)
    return features
