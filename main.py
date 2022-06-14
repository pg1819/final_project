import os
import pandas
import pickle

from collections import Counter
from dependency_tree import dependency_tree_feature_extraction, split_into_sentences
from word_embedding import split_into_paragraphs, lemmatise, word_embedding_feature_extraction
from word_distribution import word_distribution_feature_extraction


def load_model(path):
    with open(path, "rb") as target:
        model = pickle.load(target)
    return model


def detect(file, method):
    path = os.path.join("./static/uploaded_file/", file)

    if method == "word_embedding":
        yield 'data: {}\n\n'.format(0)
        print("Got here.")

        model = load_model("word_embedding/we_classifier.pickle")
        print("Got here..")

        paragraphs = split_into_paragraphs(path)
        print("Got here...")
        yield 'data: {}\n\n'.format(10)
        features_list = []
        for p in paragraphs:
            lemmas = lemmatise(p)
            if len(lemmas) > 50:
                features = word_embedding_feature_extraction(lemmas)
                means = features[0]
                variances = features[1]
                features_list.append(means + variances)
        yield 'data: {}\n\n'.format(80)
        df = pandas.DataFrame(features_list)  # Load the feature_level as a dataframe
        predictions = model.predict(df)  # Use the model to predict whether each paragraph is machine-translated
        result = Counter(predictions).most_common(1)[0][0]  # Use the most common prediction as the overall result
        yield 'data: {}\n\n'.format(100)
        yield 'data: {}\n\n'.format(result)
        os.remove(path)

    elif method == "word_distribution":
        yield 'data: {}\n\n'.format(0)
        model = load_model("word_distribution/wd_classifier.pickle")
        features = word_distribution_feature_extraction(path)
        yield 'data: {}\n\n'.format(50)
        features = [features]
        df = pandas.DataFrame(features)
        result = model.predict(df)[0]
        yield 'data: {}\n\n'.format(100)
        yield 'data: {}\n\n'.format(result)
        os.remove(path)

    elif method == "dependency_tree":
        model = load_model("dependency_tree/dt_classifier.pickle")
        sentences = split_into_sentences(path)
        features_list = []
        for s in sentences:
            if len(s.split()) > 5:
                features = dependency_tree_feature_extraction(s)
                features_list.append(features)
        df = pandas.DataFrame(features_list)  # Load the feature_level as a dataframe
        predictions = model.predict(df)  # Use the model to predict whether each paragraph is machine-translated
        result = Counter(predictions).most_common(1)[0][0]  # Use the most common prediction as the overall result
        yield 'data: {}\n\n'.format(100)
        yield 'data: {}\n\n'.format(result)
        os.remove(path)
