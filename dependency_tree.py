import spacy

from nltk import tokenize
from spacy.matcher import Matcher
from spacy.util import filter_spans

# This file contains functions that perform feature extraction for the dependency-tree method

nlp = spacy.load("en_core_web_sm")


def split_into_sentences(path):
    """
    Splits a given .txt file to a list of sentences.
    :param path: str
    :return: list <str>
    """
    with open(path, mode="r", encoding="utf-8") as f:
        text = f.read()
        text = text.replace("\n", " ")
        sentences = tokenize.sent_tokenize(text)
        return sentences


def sentence_length(sentence):
    """
    Finds the sentence length
    :param sentence: str
    :return: int
    """
    return len(sentence.split())


def unnormalised_np_length(doc):
    """
    Finds the sum length of noun phrases in a sentence
    :param doc: Doc
    :return: int
    """
    np_length = 0
    for x in doc.noun_chunks:
        np_length += len(x.text.split())
    return np_length


def unnormalised_vp_length(doc):
    """
    Finds the sum length of verb phrases in a sentence
    :param doc: Doc
    :return: int
    """
    pattern = [{'POS': 'VERB', 'OP': '?'},
               {'POS': 'ADV', 'OP': '*'},
               {'POS': 'AUX', 'OP': '*'},
               {'POS': 'VERB', 'OP': '+'}]

    matcher = Matcher(nlp.vocab)
    matcher.add("verb_phrase", [pattern])
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]

    vp_length = 0
    for word in filter_spans(spans):
        vp_length += len(word)
    return vp_length


def max_tree_depth_helper(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(max_tree_depth_helper(child, depth + 1) for child in node.children)
    else:
        return depth


def max_tree_depth(doc):
    """
    Computes the max depth of a dependency tree for a sentence
    :param doc: Doc
    :return: int
    """
    depth = [max_tree_depth_helper(sent.root, 0) for sent in doc.sents]
    return depth[0]


def dependency_tree_feature_extraction(sentence):
    doc = nlp(sentence)
    s_length = sentence_length(sentence)
    np_length = unnormalised_np_length(doc)
    vp_length = unnormalised_vp_length(doc)
    normalised_np_length = np_length / s_length
    normalised_vp_length = vp_length / s_length
    depth = max_tree_depth(doc)
    return [s_length, np_length, vp_length, normalised_np_length, normalised_vp_length, depth]
