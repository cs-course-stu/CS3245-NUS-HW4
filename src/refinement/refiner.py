#!/usr/bin/python3

import sys
import math
import nltk
import array
import heapq
import numpy as np
from indexer import Indexer
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem.porter import PorterStemmer


class QueryInfo:

    def __init__(self, query="", is_phrase=False):
        self.query = query
        self.is_phrase = is_phrase
        self.query_vector = None
        self.terms = None
        self.counts = None
        self.tokens = None

class Refiner:

    """ Refiner is a class to preprocess and define the boolean query.
        It divides the boolean query string into separate queries and
        uses the approaches of Query Expansion and Relevance Feedback
        to refine them.

    Args:
        indexer: an instance of Indexer used to load the index
        expand: boolean indicator for using Query Expansion
        feedback: boolean indicator for using Relevance Feedback
        alpha: a coefficient of Relevance Feedback
        beta: a coefficient of Releance Feedback
    """
    def __init__(self, indexer=None, expand=False, feedback=True,
                 alpha=0.9, beta=0.1):
        self.indexer = indexer
        self.expand = expand
        self.feedback = feedback
        self.alpha = alpha
        self.beta = beta

        self.stemmer = PorterStemmer()

    """ Refine the boolean query based on Relevance Feedback and Query Expansion

    Args:
        query: the boolean query string
        relevant_docs: the list of relevant docs

    Returns:
        query_infos: the list of query info for separate query
        postings_lists: the dictionary with terms to postings lists mapping
    """
    def refine(self, query, relevant_docs):
        # step 1: split the boolean query into separate queries
        query_infos = self._split_query(query)

        # step 2: expand all the single queries
        if self.expand:
            self._expand(query_infos)

        # step 3: tokenize all the single queries
        total_terms = self._tokenize(query_infos)

        # step 4: get the postings lists of total terms
        postings_lists = self.indexer.LoadTerms(total_terms)

        # step 5: construct query vector
        self._get_query_vector(query_infos, postings_lists)

        # step 6: perform Relevance Feedback
        if self.feedback:
            self._feedback(query_infos, relevant_docs, postings_lists)

        # step 7: return query info and the postings lists
        return query_infos, postings_lists

    """ Split the boolean query string into seprate quries

    Args:
        query: the boolean query string

    Returns:
        query_infos: a list of instances of QueryInfo
    """

    def _split_query(self, query):
        query_infos = []

        # step 1: split boolean query string based on 'AND'
        queries = query.split('AND')

        for q in queries:
            q = q.strip(' ')
            length = len(q)

            if length <= 0:
                continue

            query_info = None
            # step 2: create a QueryInfo class for every single query
            if q[0] == '"' and q[-1] == '"':
                if length <= 2:
                    continue

                # determine whether the split string is a phrase
                query_info = QueryInfo(q[1:-1], is_phrase=True)
            else:
                query_info = QueryInfo(q)

            # step 3: append the query infos
            query_infos.append(query_info)

        # step 4: return query_infos
        return query_infos

    def _expand(self, query_infos):
        # step 1: expand every single query by using wordnet

        pass


    """ Perform Relevance Feedback based on Rocchio Algorithm

    Args:
        query_infos: a list of instances of QueryInfo
        relevant_docs: the list of relevant docs
        postings_lists: the dictionary with terms to postings lists mapping
    """
    def _feedback(self, query_infos, relevant_docs, postings_lists):
        for query_info in query_infos:
            terms = query_info.terms
            query_vector = query_info.query_vector
            doc_vectors = defaultdict(lambda: np.zeros(len(terms)))

            # step 1: multiple vector by alpha
            query_vector *= self.alpha

            # step 2: get the doc vectors
            for i, term in enumerate(terms):
                postings_list = postings_lists[term]
                postings = postings_list[0]
                weights = postings_list[1]

                weight = 0
                for j in range(0, len(postings)):
                    doc = postings[j]

                    if doc not in relevant_docs:
                        continue

                    doc_vectors[doc][i] = weights[j]

            # step 2: normalize the doc vectors
            for doc in doc_vectors:
                doc_vector = doc_vectors[doc]
                ratio = np.linalg.norm(doc_vector) / self.beta
                doc_vector /= ratio * len(relevant_docs)
                query_vector += doc_vector

            # step 3: normalize the query vector
            query_vector /= np.linalg.norm(query_vector)

    """ Tokenize the all the queries in the query_infos

    Args:
        query_infos: a list of instances of QueryInfo that contains the query

    Returns:
        total_terms: all terms in the boolean query
    """
    def _tokenize(self, query_infos):
        total_terms = set()

        for query_info in query_infos:
            query = query_info.query

            # step 1: tokenize the query string
            tokens = [
                word for sent in nltk.sent_tokenize(query)
                for word in nltk.word_tokenize(sent)
            ]

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

            # step 6: update total_terms
            total_terms.update(terms)

        # step 7: return total_terms
        return total_terms

    """ Get the query vector based on the postings_lists

    Args:
        query_infos: the list of query info
        postings_lists: the dictionary with terms to postings lists mapping
    """
    def _get_query_vector(self, query_infos, postings_lists):
        # N = len(self.indexer.total_doc) + 1
        N = 1000

        for query_info in query_infos:
            # step 1: initlization
            length = 0
            terms = query_info.terms
            counts = query_info.counts
            query_vector = np.zeros(len(terms))

            # step 2: calculate weights by using tf-idf
            for i, term in enumerate(terms):
                tf = 1 + math.log(counts[i])
                df = len(postings_lists[term][0])
                idf = math.log(N / df) if df else 0
                weight = tf * idf

                query_vector[i] = weight
                length += weight * weight

            # step 3: normalize the query vector
            if length > 0:
                length = math.sqrt(length)

            for i in range(0, len(terms)):
                query_vector[i] /= length

            # step 4: update query_info
            query_info.query_vector = query_vector

if __name__ == '__main__':
    refiner = Refiner()

    query = '"Computer Science" AND Refiner can tokenize query strings into terms and tokens'
    terms = ['refin', 'can', 'token', 'queri', 'string', 'into', 'term', 'and', 'comput', 'scienc']
    counts = [1, 1, 2, 1, 1, 1, 1, 1]
    postings_lists = {
        'into'    : (np.array([0, 1, 3, 5]), np.array([1, 5, 6, 1]), [np.array([5, ])] ),
        'queri'   : (np.array([0, ])       , np.array([5, ])       , [np.array([3, ])] ),
        'can'     : (np.array([0, 7, 9])   , np.array([1,10, 3, 1]), [np.array([1, ])] ),
        'term'    : (np.array([0, 2, 4, 6]), np.array([1, 5, 6,10]), [np.array([6, ])] ),
        'refin'   : (np.array([0, 8])      , np.array([1, 3])      , [np.array([0, ])] ),
        'token'   : (np.array([0, 1, 4, 7]), np.array([1, 7, 6, 3]), [np.array([2, 8])]),
        'string'  : (np.array([0, 2, 5, 8]), np.array([1, 5, 6, 1]), [np.array([4, ])] ),
        'and'     : (np.array([0, 3, 6, 9]), np.array([1, 3, 6, 9]), [np.array([7, ])] ),
        'scienc'  : (np.array([0, 3, 6, 9]), np.array([1, 3, 6, 9]), [np.array([7, ])] ),
        'comput'  : (np.array([0, 3, 6, 9]), np.array([1, 3, 6, 9]), [np.array([7, ])] )
    }

    test = 'refine' if len(sys.argv) == 1 else sys.argv[1]

    if test == '_split_query':
        query_infos = refiner._split_query(query)
        for query_info in query_infos:
            print("query: ", query_info.query)
            print("is_phrase: ", query_info.is_phrase)
    elif test == '_tokenize':
        query_infos = refiner._split_query(query)
        total_terms = refiner._tokenize(query_infos)
        print(total_terms)
        for query_info in query_infos:
            print("query: ", query_info.query)
            print("is_phrase: ", query_info.is_phrase)
            print("terms: ", query_info.terms)
            print("counts: ", query_info.counts)
            print("tokens: ", query_info.tokens)
    elif test == '_get_query_vector':
        query_infos = refiner._split_query(query)
        total_terms = refiner._tokenize(query_infos)
        refiner._get_query_vector(query_infos, postings_lists)
        for query_info in query_infos:
            print("query_vector", query_info.query_vector)
    elif test == 'refine':
        query_infos, _ = refiner.refine(query, [0, 5])
        for query_info in query_infos:
            print("query: ", query_info.query)
            print("terms: ", query_info.terms)
            print("query_vector: ", query_info.query_vector)
