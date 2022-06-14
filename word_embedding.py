import itertools
import nltk
import numpy as np
import pickle
import spacy

from collections import defaultdict

# This file contains functions that perform feature extraction for the word-embedding method

nltk.download("punkt")  # For tokenising sentences
nltk.download("averaged_perceptron_tagger")  # For POS tagging words
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def split_into_paragraphs(path):
    """
    Splits a text file into list of paragraphs
    :param path: str
    :return: list <str>
    """
    with open(path, mode="r", encoding="utf-8") as f:
        text = f.read()
        paragraphs = text.split("\n\n")
        return [p.replace("\n", " ") for p in paragraphs]


def lemmatise(paragraph):
    """
    Lemmatises every word in a text into its base form
    :param paragraph: str
    :return: list <str>
    """
    doc = nlp(paragraph)
    lemmas = [str(token.lemma_).lower() for token in doc
              if token.is_alpha and not token.is_punct]
    return lemmas


def save_glove_pkl(glove_txt="word_embedding/gloVe/gloVe.6B.50d.txt", glove_pkl="word_embedding/gloVe/gloVe_dict.pkl"):
    """
    Saves the GloVe word embeddings as a dict
    :param glove_txt: str
    :param glove_pkl: str
    :return: None
    """
    embeddings_dict = {}
    with open(glove_txt, "r", encoding="utf-8") as file:
        for line in file:
            line_list = line.split()
            word = line_list[0]
            embedding = np.array(line_list[1:], dtype="float64")
            embeddings_dict[word] = embedding

    with open(glove_pkl, "wb") as f:
        pickle.dump(embeddings_dict, f)


def read_glove_pkl(glove_pkl="word_embedding/gloVe/gloVe_dict.pkl"):
    """
    Loads GloVe word embeddings pickle file into a dictionary
    :param glove_pkl: str
    :return: dict
    """
    with open(glove_pkl, "rb") as f:
        glove_dict = pickle.load(f)
    return glove_dict


def init_pos_dict():
    """
    Initialises a dictionary with pos pairs as keys
    :return: dict
    """
    pos_pairs_dict = defaultdict(list)
    pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN",
                "JJ", "JJR", "JJS", "LS", "MD", "NN",
                "NNS", "NNP", "NNPS", "PDT", "POS", "PRP",
                "PRP$", "RB", "RBR", "RBS", "RP", "SYM",
                "TO", "UH", "VB", "VBD", "VBG", "VBN",
                "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
    for tag1 in pos_tags:
        for tag2 in pos_tags:
            pos_pairs_dict[frozenset((tag1, tag2))] = []
    return pos_pairs_dict


def get_euclidean_distance(first_vector, second_vector):
    """
    Finds the euclidean distance between two vectors
    :param first_vector: numpy.ndarray <float> or NoneType
    :param second_vector: numpy.ndarray <float> or NoneType
    :return: float or NoneType
    """
    if first_vector is None or second_vector is None:
        return None

    return np.linalg.norm(first_vector - second_vector)


def get_mean(np_array):
    if len(np_array) == 0:
        return 0.0
    else:
        return np.mean(np_array)


def get_variance(np_array):
    if len(np_array) == 0:
        return 0.0
    else:
        return np.var(np_array)


def word_embedding_feature_extraction(lemmas):
    """
    Performs feature extraction on a paragraph
    :param lemmas: list <str>
    :return: list <float>, list <float>
    """
    pos_tagged_paragraph = nltk.pos_tag(lemmas)

    # Compare similarity between each pair of words and store it on the appropriate group dictionary
    pos_dict = init_pos_dict()  # Stores distance of different combinations of pos pairs
    glove_dict = read_glove_pkl()  # Stores the word embedding for each word

    for pos_pair1, pos_pair2 in itertools.combinations(pos_tagged_paragraph, 2):
        first_word = pos_pair1[0]
        second_word = pos_pair2[0]
        first_tag = pos_pair1[1]
        second_tag = pos_pair2[1]

        distance = get_euclidean_distance(glove_dict.get(first_word), glove_dict.get(second_word))

        if distance is not None:
            # If both words have the same POS tag, preserve the minimum distance
            if first_tag == second_tag:
                if not pos_dict[frozenset((first_tag, second_tag))]:
                    pos_dict[frozenset((first_tag, second_tag))].append(distance)
                elif pos_dict[frozenset((first_tag, second_tag))] > distance:
                    pos_dict[frozenset((first_tag, second_tag))] = [distance]

            # Otherwise, add the distance to the list
            else:
                pos_dict[frozenset((first_tag, second_tag))].append(distance)

    # Calculate mean and variance bt_en_fr_features.csv in each group
    mean_list = []
    variance_list = []

    for (_, distances) in sorted(pos_dict.items()):
        np_distances = np.array(distances)
        mean = get_mean(np_distances)
        variance = get_variance(np_distances)
        mean_list.append(mean)
        variance_list.append(variance)

    return [mean_list, variance_list]
