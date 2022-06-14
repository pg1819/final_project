import os
import pandas

from collections import Counter
from dependency_tree import dependency_tree_feature_extraction, split_into_sentences
from word_embedding import split_into_paragraphs, lemmatise, word_embedding_feature_extraction
from word_distribution import word_distribution_feature_extraction
from util import load_model

"""
Evaluation by document-level
"""

dt_model = load_model("dependency_tree/dt_classifier.pickle")
wd_model = load_model("word_distribution/wd_classifier.pickle")
we_model = load_model("word_embedding/we_classifier.pickle")


def dependency_tree(path, save_file):
    sentences = split_into_sentences(path)
    features_list = []
    for s in sentences:
        if len(s.split()) > 5:
            features = dependency_tree_feature_extraction(s)
            features_list.append(features)

    df = pandas.DataFrame(features_list)
    predictions = dt_model.predict(df)
    result = Counter(predictions).most_common(1)[0][0]

    with open(save_file, mode="a") as f:
        f.write(result + "\n")


def word_distribution(path, save_file):
    features = [word_distribution_feature_extraction(path)]
    df = pandas.DataFrame(features)
    result = wd_model.predict(df)[0]

    with open(save_file, mode="a") as f:
        f.write(result + "\n")


def word_embedding(path, save_file):
    paragraphs = split_into_paragraphs(path)
    features_list = []
    for p in paragraphs:
        lemmas = lemmatise(p)
        if len(lemmas) > 50:
            means, variances = word_embedding_feature_extraction(lemmas)
            features = means + variances
            if len(features) == 1332:
                features_list.append(features)

    df = pandas.DataFrame(features_list)
    predictions = we_model.predict(df)
    result = Counter(predictions).most_common(1)[0][0]

    with open(save_file, mode="a") as f:
        f.write(result + "\n")


def batch_evaluate_chapter(method, root, save_file):
    for chapter in os.scandir(root):
        for file in os.scandir(chapter.path):
            print(file.path)
            if method == "dependency_tree":
                dependency_tree(file.path, save_file)
            elif method == "word_distribution":
                word_distribution(file.path, save_file)
            elif method == "word_embedding":
                word_embedding(file.path, save_file)


batch_evaluate_chapter(root="./dataset/mt_german_chapters", method="dependency_tree",
                       save_file="results/dependency_tree/dt_german_chapter_level.txt")
batch_evaluate_chapter(root="./dataset/mt_japanese_chapters", method="dependency_tree",
                       save_file="results/dependency_tree/dt_japanese_chapter_level.txt")

batch_evaluate_chapter(root="./dataset/mt_german_chapters", method="word_distribution",
                       save_file="results/word_distribution/wd_german_chapter_level.txt")
batch_evaluate_chapter(root="./dataset/mt_japanese_chapters", method="word_distribution",
                       save_file="results/word_distribution/wd_japanese_chapter_level.txt")

batch_evaluate_chapter(root="./dataset/mt_german_chapters", method="word_embedding",
                       save_file="results/word_embedding/we_german_chapter_level.txt")
batch_evaluate_chapter(root="./dataset/mt_japanese_chapters", method="word_embedding",
                       save_file="results/word_embedding/we_japanese_chapter_level.txt")
