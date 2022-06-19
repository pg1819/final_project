import os
import pandas

from back_translation import back_translate, back_translation_feature_extraction
from dependency_tree import dependency_tree_feature_extraction, split_into_sentences
from word_embedding import split_into_paragraphs, lemmatise, word_embedding_feature_extraction
from word_distribution import word_distribution_feature_extraction
from util import load_model

"""
Evaluation by feature-level
"""

bt_model = load_model("back_translation/bt_classifier.pickle")
dt_model = load_model("dependency_tree/dt_classifier.pickle")
wd_model = load_model("word_distribution/wd_classifier.pickle")
we_model = load_model("word_embedding/we_classifier.pickle")


def back_translation(path, save_file):
    sentences = split_into_sentences(path)
    features_list = []
    for s in sentences:
        if len(s.split()) > 5:
            bt_s = back_translate(s)
            features = back_translation_feature_extraction(source_text=s, back_translation=bt_s)
            features_list.append(features)

    df = pandas.DataFrame(features_list)
    predictions = bt_model.predict(df)
    result = list(predictions)

    with open(save_file, mode="a") as f:
        for item in result:
            f.write(item + "\n")


def dependency_tree(path, save_file):
    sentences = split_into_sentences(path)
    features_list = []
    for s in sentences:
        if len(s.split()) > 5:
            features = dependency_tree_feature_extraction(s)
            features_list.append(features)
    df = pandas.DataFrame(features_list)  # Load the feature_level as a dataframe
    predictions = dt_model.predict(df)  # Use the model to predict whether each paragraph is machine-translated
    result = list(predictions)

    with open(save_file, mode="a") as f:
        for item in result:
            f.write(item + "\n")


def word_distribution(path, save_file):
    features = word_distribution_feature_extraction(path)
    df = pandas.DataFrame([features])
    result = wd_model.predict(df)[0]

    with open(save_file, mode="a") as f:
        f.write(result + "\n")


def word_embedding(path, save_file):
    paragraphs = split_into_paragraphs(path)
    features_list = []
    for p in paragraphs:
        lemmas = lemmatise(p)
        if len(lemmas) > 50:
            features = word_embedding_feature_extraction(lemmas)
            mean_list = features[0]
            variance_list = features[1]
            features = mean_list + variance_list
            if len(features) == 1332:
                features_list.append(features)

    df = pandas.DataFrame(features_list)
    predictions = we_model.predict(df)
    result = list(predictions)

    with open(save_file, mode="a") as f:
        for item in result:
            f.write(item + "\n")


def batch_evaluate_feature(method, root, save_file):
    for chapter in os.scandir(root):
        for file in os.scandir(chapter.path):
            print(file.path)
            if method == "dependency_tree":
                dependency_tree(file.path, save_file)
            elif method == "word_distribution":
                word_distribution(file.path, save_file)
            elif method == "word_embedding":
                word_embedding(file.path, save_file)


# batch_evaluate_feature(method="dependency_tree", root="./dataset/mt_german_chapters",
#                        save_file="dt_german_feature_level.txt")
# batch_evaluate_feature(method="dependency_tree", root="./dataset/mt_japanese_chapters",
#                        save_file="dt_japanese_feature_level.txt")
#
# batch_evaluate_feature(method="word_distribution", root="./dataset/mt_german_chapters",
#                        save_file="wd_german_feature_level.txt")
# batch_evaluate_feature(method="word_distribution", root="./dataset/mt_japanese_chapters",
#                        save_file="wd_japanese_feature_level.txt")
#
# batch_evaluate_feature(method="word_embedding", root="./dataset/mt_german_chapters",
#                        save_file="we_german_feature_level.txt")
# batch_evaluate_feature(method="word_embedding", root="./dataset/mt_japanese_chapters",
#                        save_file="we_japanese_feature_level.txt")
