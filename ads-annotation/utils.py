import html
import json
import re
from collections import defaultdict

from bs4 import BeautifulSoup, NavigableString
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

clean_space = re.compile("[ ][ ]+")
clean_numbers = re.compile(r"\b\d(\d*[ -]*\d*\d)*\b")
clean_mail = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-.]+\.[a-zA-Z0-9-]+)")
clean_url = url = re.compile('(https?://)?www\d{0,3}[.](?:[-\w.]|(?:%[\da-fA-F]{2}))+')
stop_words = set(stopwords.words('english'))


def load_data(filename):
    """
    Loads the input data.
    :param filename: filepath to the input data containing one json per line.
    :return: a list of jsons
    """
    with open(filename, 'r') as fin:
        data = [json.loads(line) for line in fin]
    return data


def remove_html_tags(text):
    """
    Extract text from HTML tags.
    :param text: string containing HTML tags
    :return: string in plain text
    """
    # Remove tags that are are used for style.
    regex_empty_tag = re.compile(r'</?(HTML|strong|b|div|a|em|font|span)( [^<>]*)?>')
    text = regex_empty_tag.sub(" ", text)
    soup = BeautifulSoup(text, "lxml")
    # Sometimes the tags are used to separate sentences, therefore I ad a "." if no sentence separator is in the string
    texts = [html.unescape(d.strip()) for d in soup.descendants if
             isinstance(d, NavigableString) and len(d.strip()) > 0]
    texts = [t + "." if not re.search(r'[.!?,;:]\s*$', t) else t for t in texts]
    return '\n'.join(texts)


def clean_text(text):
    """
    Remove a few entities
    :param text: string
    :return: string
    """
    text = re.sub(clean_mail, ' ', text)
    text = re.sub(clean_url, ' ', text)
    text = re.sub(clean_numbers, ' ', text)
    text = re.sub(clean_space, ' ', text)
    return text


def word_frequency(corpus):
    """
    Compute word frequencies in corpus.
    :param corpus: list of sentences
    :return: list of tuples (word, frequency)
    """
    return ngram_frequency(corpus, 1)


def ngram_frequency(corpus, n):
    """
    Compute n-gram frequencies in corpus.
    :param corpus: list of sentences
    :param n: int
    :return: list of tuples (ngram, frequency)
    """
    vec = CountVectorizer(stop_words=set(stopwords.words('english')), ngram_range=(n, n)).fit(
        corpus)
    counts = vec.transform(corpus).sum(axis=0)
    words_freq = [(word, counts[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq


def cooccurrences_frequency(corpus, window=5, min_freq=5, stopwords=None):
    """

    :param corpus: list of sentences
    :param window: int - dimension of the context before and after the target word
    :param min_freq: int - how often a co-occurrence should occur in text
    :param stopwords: list of words to be ignored from the co-occurrences
    :return:
    """
    d = defaultdict(int)

    for text in tqdm(corpus):
        tokens = word_tokenize(text)
        for i, token in enumerate(tokens):
            for j in range(i + 1, min(i + window, len(tokens))):
                if (not stopwords or (token not in stopwords) and (tokens[j] not in stopwords)) \
                        and re.match(r"\w+", token) and re.match(r"\w+", tokens[j]):
                    d[(token, tokens[j])] += 1
                    d[(tokens[j], token)] += 1

    cooccurrences = [((token1, token2), value) for (token1, token2), value in d.items() if value > min_freq]

    # These co-occurrences are used for in TextRank in the graph creation.
    # I return a dict of dicts to optimize the performances.
    list_of_cooccurrences = defaultdict(dict)
    for (token1, token2), value in tqdm(cooccurrences):
        list_of_cooccurrences[token1][token2] = value

    return list_of_cooccurrences


def find_keywords_in_text(keywords, sentences, token_pattern=r"(?u)\b\w\w+\b"):
    """
    Removes keywords that do not occur in text. i.e. the tokenizar used in TfIdfVectorizer does not respect
    the sentence boundaries, therefore, some keywords are created with information from different sentences,
    which is incorrect.
    :param keywords: list of keywords
    :param text: string - list of sentences
    :param token_pattern: regular expression denoting what constitutes a “token”
    :return:
    """
    found_keywords = []

    words = defaultdict(list)
    for keyword in keywords:
        words[keyword.split(' ')[0]].append(keyword)

    token_pattern = re.compile(token_pattern)

    for sentence in sentences:
        text_tokens = token_pattern.findall(sentence)
        text_tokens = [token.lower() for token in text_tokens]

        for start_idx, text_token in enumerate(text_tokens):
            text_token = text_token.lower()
            if text_token in words:
                for keyword in words[text_token]:
                    words_keyword = keyword.split(" ")

                    end_idx = start_idx
                    is_keyword = True

                    # accept stopwords inside keywords
                    for word_keyword in words_keyword:
                        while (end_idx < len(text_tokens)) and (text_tokens[end_idx] in stop_words):
                            end_idx += 1
                        if (end_idx < len(text_tokens)) and (text_tokens[end_idx] == word_keyword):
                            end_idx += 1
                        else:
                            is_keyword = False
                            break

                    if is_keyword:
                        found_keywords.append(' '.join(text_tokens[start_idx:end_idx]))

    return found_keywords


def measure(keywords, ground_truth):
    """
    Measure for keyword extraction.
    :param keywords: set of extracted keywords
    :param ground_truth: set of ground truth keywords
    :return: precision - float, recall - float, f score - float, list of strings
    """
    def get_ngrams(keywords):
        list_ngrams = []
        for n in range(1, 4):
            for keyword in keywords:
                list_ngrams.extend(ngrams(keyword.split(), n))
        list_ngrams = [' '.join(ngrm) for ngrm in list_ngrams]
        return list_ngrams

    items_keywords = set(get_ngrams(keywords))
    items_ground_truth = set(get_ngrams(ground_truth))
    intersection = items_keywords.intersection(items_ground_truth)
    prec = float(len(intersection)) / len(items_keywords)
    rec = float(len(intersection)) / len(items_ground_truth)
    f = 2 * prec * rec / (prec + rec)
    return prec, rec, f, intersection
