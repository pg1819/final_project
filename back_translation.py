from nltk import tokenize
from nltk.translate.bleu_score import sentence_bleu
from transformers import FSMTForConditionalGeneration, FSMTTokenizer


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


def translate(input, mname):
    """
    Translates a sentence using the transformer model to the desired target language
    :param input:  str
    :param mname: str
    :return: str
    """
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)
    input_ids = tokenizer.encode(input, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


def back_translate(text):
    """
    Performs back translation on a sentence from English to German, back to English
    :param text: str
    :return: str
    """
    first_translation = translate(input=text, mname="facebook/wmt19-en-de")  # translate text to german
    back_translation = translate(input=first_translation,
                                 mname="facebook/wmt19-de-en")  # translate text back to english
    return back_translation


def back_translation_feature_extraction(source_text, back_translation):
    """
    Calculates the individual and cumulative n-gram scores for a sentence and its back-translation
    :param source_text: str
    :param back_translation: str
    :return: list <float>
    """

    # Split into array of words by spaces
    source_text_array = [source_text.split()]
    back_translation_array = back_translation.split()

    # Compute individual n-grams from 1 to 4
    f1 = sentence_bleu(source_text_array, back_translation_array, weights=(1, 0, 0, 0))
    f2 = sentence_bleu(source_text_array, back_translation_array, weights=(0, 1, 0, 0))
    f3 = sentence_bleu(source_text_array, back_translation_array, weights=(0, 0, 1, 0))
    f4 = sentence_bleu(source_text_array, back_translation_array, weights=(0, 0, 0, 1))

    # Compute cumulative n-grams from 2 to 4
    f5 = sentence_bleu(source_text_array, back_translation_array, weights=(0.5, 0.5, 0, 0))
    f6 = sentence_bleu(source_text_array, back_translation_array, weights=(0.33, 0.33, 0.33, 0))
    f7 = sentence_bleu(source_text_array, back_translation_array, weights=(0.25, 0.25, 0.25, 0.25))
    return [f1, f2, f3, f4, f5, f6, f7]

