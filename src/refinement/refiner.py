#!/usr/bin/python3

import sys
import math
import nltk
import array
import heapq
import numpy as np
# from indexer import Indexer
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem.porter import PorterStemmer

class QueryInfo:

    def __init__(self, query="", is_phrase=False):
        self.query = query
        self.is_phrase = is_phrase
        self.query_vec = None
        self.terms = None
        self.counts = None
        self.tokens = None

class Refiner:

    def __init__(self, indexer=None, alpha=0.1, beta=0.1):
        self.indexer = indexer
        self.alpha = alpha
        self.beta = beta

        self.stemmer = PorterStemmer()

    def refine(self, query, relevant_docs):
        # step 1: split the boolean query into single query
        query_infos = self.split_query(self, query)

        # step 2: extend all the single query

        # step 3: perform Relevant Feedback

        # step 4: return query info

        pass

    """ Split the boolean query string into seprate quries

    Args:
        query: the boolean query string

    Returns:
        result: a list of instances of QueryInfo
    """
    def split_query(self, query):
        query_infos = []

        # step 1: split boolean query string based on 'AND'
        queries = query.split('AND')

        for q in queries:
            q = q.strip(' ')
            length = len(q)

            if length <= 0:
                continue;

            query_info = None
            # step 2: create a QueryInfo class for every single query
            if q[0] == '"' and q[-1] == '"':
                if length <= 2:
                    continue

                # determine whether the split string is a phrase
                query_info = QueryInfo(q[1:-1], is_phrase=True)
            else:
                query_info = QueryInfo(q)

            print(query_info.query)

            # step 3: tokenize every single query
            self.tokenize(query_info)

            # step 4: append the query infos
            query_infos.append(query_info)

        # step 4: return query_infos
        return query_infos


    def extend(self, query_infos):
        # step 1: extend every single query by using wordnet

        pass

    def feedback(self, query_infos, relevant_docs):
        # step 1: get doc vectors of the relevant docs

        # step 2: perform feedback for every single query

        pass


    """ Tokenize the query in the query_info

    Args:
        query_info: an instance of QueryInfo that contains the query
    """
    def tokenize(self, query_info):
        query = query_info.query

        # step 1: tokenize the query string
        tokens = [word for sent in nltk.sent_tokenize(query)
                  for word in nltk.word_tokenize(sent)]

        # step 2: stem the tokens
        tokens = [self.stemmer.stem(token.lower()) for token in tokens]

        # step 3: get the term count
        term_count = defaultdict(lambda: 0)
        for token in tokens:
            term_count[token] += 1

        # step 4: get terms and counts
        terms = []
        counts = []
        for term in term_count:
            terms.append(term)
            counts.append(term_count[term])

        # step 5: update query info
        query_info.terms = terms
        query_info.counts = counts
        query_info.tokens = tokens

    def get_doc_vec(self):
        # step 1: look up index to get the doc vec

        pass

if __name__ == '__main__':
    refiner = Refiner()

    query = '"computer sciense" AND hello world hello python'

    test = 'split'

    if test == 'split':
        query_infos = refiner.split_query(query)
        for query_info in query_infos:
            print("query: ", query_info.query)
            print("is_phrase: ", query_info.is_phrase)
            print("terms: ", query_info.terms)
            print("counts: ", query_info.counts)
            print("tokens: ", query_info.tokens)
