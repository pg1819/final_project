import os
import pandas
import pickle

from collections import Counter
from dependency_tree import dependency_tree_feature_extraction, split_into_sentences
from highlighter import open_pdf, highlight_pdf, save_pdf, txt_to_pdf
from word_embedding import split_into_paragraphs, lemmatise, word_embedding_feature_extraction
from word_distribution import word_distribution_feature_extraction


def load_model(path):
    with open(path, "rb") as target:
        model = pickle.load(target)
    return model


def detect(file, method):
    txt_path = os.path.join("./static/uploaded_file", file)
    pdf_path = os.path.join("./static/results", os.path.splitext(file)[0] + ".pdf")
    highlighted_pdf_path = "./static/results/result.pdf"

    if os.path.isfile(highlighted_pdf_path):
        os.remove(highlighted_pdf_path)

    with open(txt_path, mode="r", encoding="utf-8-sig") as f:
        text = f.read()
        txt_to_pdf(text, pdf_path)

    if method == "dependency_tree":
        yield 'data: {}\n\n'.format(0)
        model = load_model("dependency_tree/dt_classifier.pickle")
        sentences = split_into_sentences(txt_path)
        total_sentences = []
        highlighted_sentences = []
        predictions = []
        yield 'data: {}\n\n'.format(10)
        for s in sentences:
            if len(s.split()) > 5:
                features = dependency_tree_feature_extraction(s)
                df = pandas.DataFrame([features])
                prediction = model.predict(df)[0]
                predictions.append(prediction)
                total_sentences.append(s)
                if prediction == "machine-translated":
                    highlighted_sentences.append(s)
        yield 'data: {}\n\n'.format(80)
        result = Counter(predictions).most_common(1)[0][0]
        doc = open_pdf(pdf_path)
        for hs in highlighted_sentences:
            doc = highlight_pdf(doc, hs)
        save_pdf(doc, highlighted_pdf_path)
        yield 'data: {}\n\n'.format(100)
        confidence_level = str(round((len(highlighted_sentences) / len(total_sentences)) * 100)) + " %"
        yield 'data: {}\n\n'.format(result + "," + str(confidence_level))
        doc.close()
        os.remove(txt_path)
        os.remove(pdf_path)
        print("Removed..")

    elif method == "word_embedding":
        yield 'data: {}\n\n'.format(0)
        model = load_model("word_embedding/we_classifier.pickle")
        paragraphs = split_into_paragraphs(txt_path)
        total_paragraphs = []
        highlighted_paragraphs = []
        predictions = []
        yield 'data: {}\n\n'.format(10)
        for p in paragraphs:
            lemmas = lemmatise(p)
            if len(lemmas) > 50:
                features = word_embedding_feature_extraction(lemmas)
                means = features[0]
                variances = features[1]
                df = pandas.DataFrame([means + variances])
                prediction = model.predict(df)[0]
                predictions.append(prediction)
                total_paragraphs.append(p)
                if prediction == "machine-translated":
                    highlighted_paragraphs.append(p)
        yield 'data: {}\n\n'.format(80)
        result = Counter(predictions).most_common(1)[0][0]
        doc = open_pdf(pdf_path)
        for hp in highlighted_paragraphs:
            doc = highlight_pdf(doc, hp)
        save_pdf(doc, highlighted_pdf_path)
        print(len(highlighted_paragraphs))
        print(len(total_paragraphs))
        print(predictions)
        confidence_level = str(round((len(highlighted_paragraphs) / len(total_paragraphs)) * 100)) + " %"
        yield 'data: {}\n\n'.format(100)
        yield 'data: {}\n\n'.format(result + "," + str(confidence_level))
        doc.close()
        os.remove(txt_path)
        os.remove(pdf_path)
        print("Removed..")

    elif method == "word_distribution":
        yield 'data: {}\n\n'.format(0)
        model = load_model("word_distribution/wd_classifier.pickle")
        confidence_level = "N/A"
        yield 'data: {}\n\n'.format(10)
        features = word_distribution_feature_extraction(txt_path)
        features = [features]
        yield 'data: {}\n\n'.format(50)
        df = pandas.DataFrame(features)
        result = model.predict(df)[0]
        doc = open_pdf(pdf_path)
        if result == "machine_translated":
            with open(file, mode="r", encoding="utf-8-sig") as f:
                content = f.read()
                doc = highlight_pdf(doc, content)
        save_pdf(doc, highlighted_pdf_path)
        yield 'data: {}\n\n'.format(100)
        yield 'data: {}\n\n'.format(result + "," + str(confidence_level))
        doc.close()
        os.remove(txt_path)
        os.remove(pdf_path)
        print("Removed..")
