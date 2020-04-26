#!/usr/bin/python3
import re
import nltk
import sys
import getopt
from searcher import Searcher

expand = False
feedback = True
score = False

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results\n")
    print("examples: python search.py -d dictionary.txt -p postings.txt -q q1.txt -o q1.out\n")
    print("options:\n"
          "  -d  dictionary file path\n"
          "  -p  postings file path\n"
          "  -q  queries file path\n"
          "  -o  search results file path\n"
          "  -e  enable query expansion\n"
          "  -f  disable relevance feedback\n"
          "  -s  enable printing score\n")

def run_search(dict_file, postings_file, queries_file, results_file, expand, feedback, score):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    # This is an empty method
    # Pls implement your code in below

    searcher = Searcher(dict_file, postings_file, expand=expand, feedback=feedback, score=score)

    first_line = True
    with open(queries_file, 'r') as fin, \
         open(results_file, 'w') as fout:

        query = None
        relevant_docs = []
        for line in fin:
            if first_line:
                query = line
                first_line = False
            elif line not in ['\r', '\n', '\r\n']:
                relevant_docs.append(int(line))

        result, score = searcher.search(query, relevant_docs)
        result = map(str, result)

        result = '\n'.join(result)
        fout.write(result)

        if score:
            scores = '\n' + ' '.join(map(str, scores))
            fout.write(scores)

dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    elif o == '-e':
        expand = True
    elif o == '-f':
        feedack = False
    elif o == '-s':
        score = True
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output, expand, feedback, score)
