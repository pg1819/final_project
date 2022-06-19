import csv
import os

import pandas
import pickle

from back_translation import back_translation_feature_extraction, back_translate
from dependency_tree import split_into_sentences, dependency_tree_feature_extraction
from word_distribution import word_distribution_feature_extraction
from word_embedding import split_into_paragraphs, word_embedding_feature_extraction, lemmatise

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# This file contains utility functions used for performing the experiments


def save_features(method, features, result, save_path):
    """
    Saves the feature vector to a csv file
    :param method: str
    :param features: list<>
    :param result: str
    :param save_path: str
    :return: None
    """
    if method == "word_distribution":
        slope = features[0]
        r2 = features[1]
        mse = features[2]
        if round(slope, 2) != 0.0:
            with open(save_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([slope, r2, mse, result])

    elif method == "word_embedding":
        with open(save_path, "a", newline="") as f:
            mean_list = features[0]
            variance_list = features[1]
            writer = csv.writer(f)
            row = mean_list + variance_list + [result]
            writer.writerow(row)

    else:
        with open(save_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = features + [result]
            writer.writerow(row)


def save_model(method):
    if method == "back_translation":
        dataset = pandas.read_csv("back_translation/bt_en_fr_features.csv", on_bad_lines="skip")
        features = dataset.iloc[:, :-1]
        result = dataset.iloc[:, 7]
        model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        model.fit(features, result)
        with open("back_translation/bt_classifier.pickle", "wb") as f:
            pickle.dump(model, f)

    elif method == "dependency_tree":
        dataset = pandas.read_csv("dependency_tree/dt_en_fr_features.csv", on_bad_lines="skip")
        features = dataset.iloc[:, :-1]
        result = dataset.iloc[:, 6]
        model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        model.fit(features, result)
        with open("dependency_tree/dt_classifier.pickle", "wb") as f:
            pickle.dump(model, f)

    elif method == "word_distribution":
        dataset = pandas.read_csv("word_distribution/wd_en_fr_features.csv", on_bad_lines="skip")
        features = dataset.iloc[:, :-1]
        result = dataset.iloc[:, 3]
        model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        model.fit(features, result)
        with open("word_distribution/wd_classifier.pickle", "wb") as f:
            pickle.dump(model, f)

    elif method == "word_embedding":
        dataset = pandas.read_csv("word_embedding/we_en_fr_features.csv", on_bad_lines="skip")
        features = dataset.iloc[:, :-1]
        result = dataset.iloc[:, 1332]
        model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        model.fit(features, result)
        with open("word_embedding/we_classifier.pickle", "wb") as f:
            pickle.dump(model, f)


def evaluate(filename, method):
    dataset = pandas.read_csv(filename, on_bad_lines='skip')
    features = dataset.iloc[:, :-1]

    if method == "back_translation":
        result = dataset.iloc[:, 7]

    elif method == "dependency_tree":
        result = dataset.iloc[:, 6]

    elif method == "word_distribution":
        result = dataset.iloc[:, 3]

    elif method == "word_embedding":
        result = dataset.iloc[:, 1332]

    # Split dataset into training and testing set
    features_train, features_test, result_train, result_test = train_test_split(features, result, test_size=0.20)

    # Train SVM with SGD
    model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    model.fit(features_train, result_train)
    pred = model.predict(features_test)

    # Perform evaluation
    print(confusion_matrix(result_test, pred))
    print("")
    print(classification_report(result_test, pred))
    print("")
    print(accuracy_score(result_test, pred))


def batch_save_features(method, root, result, save_path):
    """
    Performs feature extraction for all the dataset, saving to its appropriate csv file
    :param method: str
    :param root: str
    :param result: str
    :param save_path: str
    :return: None
    """
    for sub_dir in os.scandir(root):
        for file in os.scandir(sub_dir.path):
            print(file.path)

            if method == "back_translation":
                sentences = split_into_sentences(file.path)
                for s in sentences:
                    if len(s.split()) > 5:
                        bt_s = back_translate(s)
                        features = back_translation_feature_extraction(source_text=s, back_translation=bt_s)
                        save_features(method, features, result, save_path)
                        break

            elif method == "dependency_tree":
                sentences = split_into_sentences(file.path)
                for s in sentences:
                    if len(s.split()) > 5:
                        features = dependency_tree_feature_extraction(s)
                        save_features(method, features, result, save_path)

            elif method == "word_distribution":
                features = word_distribution_feature_extraction(file.path)
                save_features(method, features, result, save_path)

            elif method == "word_embedding":
                paragraphs = split_into_paragraphs(file.path)
                for p in paragraphs:
                    lemmas = lemmatise(p)
                    if len(lemmas) > 50:
                        features = word_embedding_feature_extraction(lemmas)
                        save_features(method, features, result, save_path)


def recall(path):
    total = 0
    machine_translated = 0
    for row in open(path):
        if row == "machine-translated\n":
            machine_translated += 1
        total += 1
    print("Machine Translated", machine_translated)
    print("Total", total)
    print("Recall", machine_translated / total)


def load_model(path):
    with open(path, "rb") as target:
        model = pickle.load(target)
    return model

# batch_save_features(method="back_translate", root="dataset/english_chapters", result="human-written",
#                     save_path="back_translation/bt_en_fr_features.csv")
# batch_save_features(method="back_translate", root="dataset/mt_french_chapters", result="machine-translated",
#                     save_path="back_translation/bt_en_fr_features.csv")
#
# batch_save_features(method="dependency_tree", root="dataset/english_chapters", result="human_written",
#                     save_path="dependency_tree/dt_en_fr_features.csv")
# batch_save_features(method="dependency_tree", root="dataset/mt_french_chapters", result="machine-translated",
#                     save_path="dependency_tree/dt_en_fr_features.csv")
#
# batch_save_features(method="word_distribution", root="dataset/english_chapters", result="human-written",
#                     save_path="word_distribution/wd_en_fr_features.csv")
# batch_save_features(method="word_distribution", root="dataset/mt_french_chapters", result="machine-translated",
#                     save_path="word_distribution/wd_en_fr_features.csv")
#
# batch_save_features(method="word_embedding", root="dataset/english_chapters", result="human-written",
#                     save_path="word_embedding/we_en_fr_features.csv")
# batch_save_features(method="word_embedding", root="dataset/mt_french_chapters", result="machine-translated",
#                     save_path="word_embedding/we_en_fr_features.csv")

# evaluate(filename="./back_translation/bt_en_fr_features.csv", method="back_translation")
