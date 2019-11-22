"""Python implementation of the TextRank algoritm.

From this paper:
    https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

Based on:
    https://gist.github.com/voidfiles/1646117
    https://github.com/davidadamojr/TextRank
"""
import math
import operator

import networkx as nx
import nltk


def setup_environment():
    """Download required resources."""
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    print('Completed resource downloads.')


def filter_for_tags(tagged, tags=['NN', 'JJ', 'NNP', 'NNS', 'NNPS']):
    """Apply syntactic filters based on POS tags."""
    return [item for item in tagged if item[1] in tags]


def normalize(tagged):
    """Return a list of tuples with the first item's periods removed."""
    return [(item[0].replace('.', ''), item[1]) for item in tagged]


def build_graph(nodes, edges_score):
    """Return a networkx graph instance.

    :param nodes: List of hashables that represent the nodes of a graph.
    """
    gr = nx.Graph()  # initialize an undirected graph
    gr.add_nodes_from(nodes)
    # silvia
    for i, node in enumerate(nodes):
        for j in range(i+1, len(nodes)):
            weight = edges_score(node, nodes[j])
            if weight:
                gr.add_edge(node, nodes[j], weight=weight)

    return gr


def extract_key_phrases(corpus, edges_score, max_words=1):
    """Return a set of key phrases.

    :param text: A string.
    """
    word_tokens = nltk.word_tokenize(corpus)

    tagged = nltk.pos_tag(word_tokens)
    textlist = [x[0] for x in tagged]

    tagged = filter_for_tags(tagged)
    tagged = normalize(tagged)

    unique_word_set = set([x[0] for x in tagged])
    word_set_list = list(unique_word_set)

    graph = build_graph(word_set_list, edges_score)

    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank.items(), key=operator.itemgetter(1),
                        reverse=True)
    # the number of keyphrases returned will be relative to the size of the
    # text (a third of the number of vertices)
    one_third = len(word_set_list) // 3
    keyphrases = dict(keyphrases[0:one_third + 1])

    # take keyphrases with multiple words into consideration as done in the
    # paper - if two words are adjacent in the text and are selected as
    # keywords, join them together
    modified_key_phrases = set([])
    # keeps track of individual keywords that have been joined to form a
    # keyphrase
    dealt_with = set([])

    # silvia
    for dimension in range(max_words, 0, -1):
        for i in range(len(textlist) - dimension + 1):
            # create keyphrases only if all the words have been choose as keyphrases and
            # it contains at least one word that was not dealt with
            if all(word in keyphrases for word in textlist[i: i + dimension]) and any(
                    word not in dealt_with for word in textlist[i: i + dimension]):
                score = sum(keyphrases[word] for word in textlist[i: i + dimension])
                modified_key_phrases.add((' '.join(textlist[i: i + dimension]), score))
                dealt_with.update(textlist[i: i + dimension])

    return sorted(modified_key_phrases, key=operator.itemgetter(1), reverse=True)
