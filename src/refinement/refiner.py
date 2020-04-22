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
        self.query = ""
        self.is_phrase=False
        self.query_vec = None
        self.terms = None
        self.counts = None
        self.tokens = None

class Refiner:

    def __init__(self, indexer, alpha=0.1, beta=0.1):
        self.indexer = indexer
        self.alpha = alpha
        self.beta = beta

    def refine(self, query, relevant_docs):
        # step 1: split the boolean query into single query

        # step 2: extend all the single query

        # step 3: perform Relevant Feedback

        # step 4: return query info

        pass

    def split_query(self, query):
        # step 1: split the boolean query

        # step 2: create a QueryInfo class for every single query

        # step 3: tokenize every single query

        pass

    def extend(self, query_infos):
        # step 1: extend every single query by using wordnet

        pass

    def feedback(self, query_infos, relevant_docs):
        # step 1: get doc vectors of the relevant docs

        # step 2: perform feedback for every single query

        pass

    def tokenize(self, query_info):
        # follow homework 3
        pass

    def split_query(self, query):
        # step 1: split boolean query string based on 'AND'

        # step 2: determine whether the split string is a phrase

        # step 3: create a QueryInfo for every single query string

        pass

    def get_doc_vec(self):
        # step 1: look up index to get the doc vec

        pass
