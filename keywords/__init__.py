import operator
import os
import pickle
import re

import RAKE
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from keywords import textrank
from utils import stop_words, cooccurrences_frequency


class KeywordsTfidf:
    """
    Class that extracts the most meaningful keywords from a give text, based on the Tf-Idf score of terms.
    """
    FILE_IDF_SCORES = 'fixtures/tfidf_vectorizer_counts.pickle'
    tfidf_vectorizer = None
    counts = None

    def load_model(self):
        """
        Load the IDF counts from the serialized model
        :return:
        """
        if not os.path.exists(self.FILE_IDF_SCORES):
            raise FileNotFoundError("The file with the idf scores was not found! Have you computed them?")

        with open(self.FILE_IDF_SCORES, 'rb') as handle:
            self.tfidf_vectorizer, self.counts = pickle.load(handle)

    def compute_idf(self, corpus, serialize=True):
        """
        Using the given corpus, compute the IDF scores for all words, bigrams and trigrams from corpus
        that occur at least 10 times
        :param corpus: a list of strings, each string representing a document
        :param serialize: If set to True, the file will be serialized for later reuse
        :return:
        """
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=stop_words, token_pattern=r"(?u)\b\w\w+\b",
                                                ngram_range=(1, 3),
                                                min_df=10)
        self.counts = self.tfidf_vectorizer.fit(corpus)

        if serialize:
            self._serialize()

    def _serialize(self):
        """
        Serialize the TfidfVectorizer model and the IDF counts
        :return:
        """
        with open(self.FILE_IDF_SCORES, 'wb') as handle:
            pickle.dump((self.tfidf_vectorizer, self.counts), handle)

    def get_keywords(self, text):
        """
        Extract the keywords using TF-IDF scores
        :param text: the from where the keywords are extracted
        :return: a list of keywords
        """
        if not os.path.exists(KeywordsTfidf.FILE_IDF_SCORES):
            raise FileNotFoundError("The file with the idf scores was not found! Have you computed them?")

        # load the model if it is not in memory.
        if not self.tfidf_vectorizer:
            self.load_model()

        # transform the ad in a vector of TFIDF scores
        ads_tfidf = self.tfidf_vectorizer.transform([text]).toarray()
        keywords = [(word, value) for value, word in zip(ads_tfidf[0], self.tfidf_vectorizer.get_feature_names()) if
                    value > 0]

        # sort the keywords and keep only the top ones.
        # I chose to return topn keyqords, where topn is 10% of the total number of words in text.
        # A detailed analysis is needed to find the best parameter
        keywords = sorted(keywords, key=operator.itemgetter(1), reverse=True)
        topn = max(int(sum(1 for _ in re.finditer(self.tfidf_vectorizer.token_pattern, text)) * 0.1), 3)
        selected_keywords = KeywordsTfidf._filter_keywords(keywords[:topn])
        return selected_keywords

    @staticmethod
    def _filter_keywords(keywords):
        i = 0
        dealt_with = set([])
        new_keys = []
        while i < len(keywords) - 1:
            keys = []
            for j in range(i, len(keywords)):
                if keywords[j][1] == keywords[i][1]:
                    keys.append(keywords[j][0])
                else:
                    break
            keys = sorted(keys, key=lambda x: len(x), reverse=True)
            for key in keys:
                if any(word not in dealt_with for word in key.split(' ')):
                    dealt_with.update(key.split(' '))
                    new_keys.append(key)

            i = j
        return new_keys


class KeywordsRake:
    """
    Class that extracts the most meaningful keywords from a give text, using the Rake algorithm.
    """
    Rake = RAKE.Rake(RAKE.SmartStopList())

    def get_keywords(self, text):
        """
        Extract keywords from text using Rake algorithm
        :param text: input string
        :return: a list of keywords
        """
        # extract keywords that have up to three words and that occur at least one in the text
        keywords = self.Rake.run(text, maxWords=3, minFrequency=1)
        # I chose to return topn keywords, where topn is 10% of the total number of words in text.
        # A detailed analysis is needed to find the best parameter
        topn = max(int(len(text.split(' ')) * 0.1), 3)
        return keywords[:topn]


class KeywordsTextRank:
    """
    Class that extracts the most meaningful keywords from a give text, using the TextRank algorithm.
    """
    FILE_COOCCURRENCES = 'fixtures/co-occurrences.pickle'
    co_occurrences = None

    def load_model(self):
        """
        Load the co-occurrences counts from a serialized model
        :return:
        """
        if not os.path.exists(KeywordsTextRank.FILE_COOCCURRENCES):
            raise FileNotFoundError("The file with the co-occurrences was not found! Have you computed them?")

        with open(KeywordsTextRank.FILE_COOCCURRENCES, 'rb') as handle:
            KeywordsTextRank.co_occurrences = pickle.load(handle)

    @staticmethod
    def compute_coocurrences(corpus):
        """
        Compute co-occurrences for TextRank.
        :param corpus: a list of strings
        :return:
        """
        cooccurrences = cooccurrences_frequency(corpus, stopwords=stopwords.words("english"))
        with open(KeywordsTextRank.FILE_COOCCURRENCES, 'wb') as handle:
            pickle.dump(cooccurrences, handle)

    def get_keywords(self, ad):
        """

        :param ad: Extract keywords from text using TextRank algorithm
        :return: a list of keywords
        """
        if not self.co_occurrences:
            self.load_model()

        keywords = textrank.extract_key_phrases(ad, KeywordsTextRank.get_scores, max_words=3)
        return keywords

    @staticmethod
    def get_scores(word1, word2):
        """
        Get a score for the co-occurrence of word1 and word2 in corpus. It is used inside the TextRank algorithm.
        :param word1: string representing a word from corpus
        :param word2: string representing a word from corpus
        :return:
        """
        return KeywordsTextRank.co_occurrences[word1][word2] if word1 in KeywordsTextRank.co_occurrences and word2 in \
                                                                KeywordsTextRank.co_occurrences[word1] else 0


class GloveFilter:
    """
    Class that handles GloVe word vectors.
    """
    FILE_SKILLS = "fixtures/skills.txt"
    GLOVE_VECTORS = "fixtures/gensim_glove_vectors.txt"
    glove_model = None

    def __init__(self):
        GloveFilter.glove_model = KeyedVectors.load_word2vec_format(GloveFilter.GLOVE_VECTORS, binary=False)

    @staticmethod
    def initialize(glove_file, vocabulary):
        """
        Pre-processes the original GloVe file and creates a new file using only the vectorial representations
        for the vectors from our corpus.
        :param glove_file: filepath to any GloVe representation downloaded from
                https://nlp.stanford.edu/projects/glove/, it has to be a .txt file
        :param vocabulary_file: file with one word per line
        :return:
        """

        # to load the verorial representation I used Gensim framework.
        # The following 2 methods extract information necessary to write the Gensim format
        def get_glove_info(glove_file_name, vocabulary=None):
            """
            Counts the number of vectors and their dimension that will be used
            :param glove_file: filepath to any GloVe representation downloaded from
                https://nlp.stanford.edu/projects/glove/, it has to be a .txt file
            :param vocabulary: file with one word per line
            :return: number of lines, the dimension of the vectors
            """
            with open(glove_file_name, 'rb') as f:
                num_lines = sum(1 for line in tqdm(f) if not vocabulary or line[:line.index(b' ')].decode('utf-8') in vocabulary)
            with open(glove_file_name, 'rb') as f:
                num_dims = len(f.readline().split()) - 1
            return num_lines, num_dims

        def glove2word2vec(glove_input_file, vocabulary=None):
            """
            Rewrites the vectors in the Gensim format.
            :param glove_input_file: filepath to any GloVe representation downloaded from
                https://nlp.stanford.edu/projects/glove/, it has to be a .txt file
            :param vocabulary: file with one word per line
            :return:
            """
            num_lines, num_dims = get_glove_info(glove_input_file, vocabulary=vocabulary)
            with open(GloveFilter.GLOVE_VECTORS, 'wb') as fout:
                fout.write("{0} {1}\n".format(num_lines, num_dims).encode('utf-8'))
                with open(glove_input_file, 'rb') as fin:
                    for line in tqdm(fin):
                        if not vocabulary or line[:line.index(b' ')].decode('utf-8') in vocabulary:
                            fout.write(line)
            return num_lines, num_dims

        glove2word2vec(glove_input_file=glove_file, vocabulary=vocabulary)

    def vector_representation(self, multiword):
        """
        Returns the representation of a multiwords
        :param multiword: input string
        :return: np.array that is the vectorial representation of the multiword.
        """
        v = []
        mw = multiword.split(' ')
        for word in mw:
            if word in GloveFilter.glove_model:
                v.append(GloveFilter.glove_model[word])
            else:
                # if any word from a multi word expression doesn't have a representation in GloVe,
                # the multi-word expression cannot be represented, so we return None.
                return None

        return np.array(v).mean(axis=0)

    def _get_target_vector(self):
        """
        Returns a vector that represents the target information, skill and requirements, from a list of skills.
        :return: np.array
        """
        with open(self.FILE_SKILLS, "r") as handle:
            all_skills = [self.vector_representation(line.lower().strip()) for line in handle]
        all_skills = [v for v in all_skills if v is not None]
        return np.array(all_skills).mean(axis=0)

    def filter_keywords(self, keywords, topn=10):
        """
        Returns a set of keywords ordered by the similarity with a target vector, the skills vector
        :param keywords: candidate keywords
        :return: list of keywords
        """
        # obtain the target vector
        skills_vector = self._get_target_vector()
        # compute the vectorial representation for each keyword
        vectors = [(key, self.vector_representation(key)) for key in keywords]
        # filter out those keywords that do not have a representation
        vectors = [vector for vector in vectors if vector[1] is not None]
        keywords, vectors = zip(*vectors)
        vectors = np.array(vectors)
        # compute similarities
        similarities = GloveFilter.glove_model.cosine_similarities(skills_vector, vectors)
        top_keywords = sorted(zip(keywords, similarities), key=operator.itemgetter(1), reverse=True)[:topn]
        return top_keywords
