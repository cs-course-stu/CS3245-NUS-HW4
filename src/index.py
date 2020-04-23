#!/usr/bin/python3
import math
import re
import os
import sys
import getopt
import csv
import numpy as np
import pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

DOC_ID = "document_id"
TITLE = "title"
CONTENT = "content"
DATE = "date_posted"
COURT = "court"


def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")


def generate_court_field(dataset_dict):
    """
    Method to generate the postings list of documents for the court field. key is
    court name and value is the list of documents that are in court.
    """

    # Store court as field
    court_field_dict = {}

    # For each document
    for doc_key in dataset_dict:

        # Get court name and make it key of court dict
        court_name = dataset_dict[doc_key][COURT]

        # Append doc_id in contents
        if court_name not in court_field_dict:
            court_field_dict[court_name] = list()
        court_field_dict[court_name].append(doc_key)

    # sort doc ids in value
    for key in court_field_dict:
        court_field_dict[key] = sorted(court_field_dict[key])

    return court_field_dict


def get_date_from_title(title):
    """
    Method to extract year from title
    """
    terms = title.split()
    brac_year = terms[-3]
    list_year = []
    for char in brac_year:
        if char != "[" and char != "]":
            list_year.append(char)

    year = "".join(list_year)
    return year


def generate_date_field(data_dict):
    """
    Method to generate the postings list of documents for the court field. key is
    court name and value is the list of documents that are in court.
    """

    # Store year from title as field
    date_dictionary = {}
    for doc_key in data_dict:

        # Get title field
        title = data_dict[doc_key][TITLE]
        year = get_date_from_title(title)
        # Append year in contents
        if year not in date_dictionary:
            date_dictionary[year] = list()
        date_dictionary[year].append(doc_key)

    # sort doc ids in value
    for key in date_dictionary:
        date_dictionary[key] = sorted(date_dictionary[key])

    return date_dictionary


def update_postings(doc_id, document_term_frequency, postings, words, is_title):
    """
    Method parses words in given sentence and updates postings list 
    """
    # Deciding suffix for zones
    if is_title:
        toAppend = ".title"
    else:
        toAppend = ".content"

    # Initialise PorterStemmer
    ps = PorterStemmer()
    
    for i in range(len(words)):
        # Pre-process word
        preprocessed_word = ps.stem(words[i]).lower()
        current_word = preprocessed_word + toAppend
        # current_word = words[i].append(toAppend)

        if current_word not in postings:
            postings[current_word] = []
            postings[current_word].append([])
            postings[current_word].append([])
            postings[current_word].append({})
        if current_word not in postings[current_word][0]:
            postings[current_word][0].append(doc_id)
            postings[current_word][1].append(None)
            postings[current_word][2][doc_id] = []
        postings[current_word][2][doc_id].append(i)
        if current_word not in document_term_frequency:
            document_term_frequency[current_word] = 0
        document_term_frequency[current_word] += 1


def normalize_weighted_tf(term_frequency, postings, doc_id):
    """
    normalize weighted term frequency by dividing all term frequency
    with square root(sum of all(square(each term frequency)))
    """
    to_be_divided_by = 0

    # Obtain sum of the square of all term frequency
    for word in term_frequency:
        term_frequency[word] = math.log(term_frequency[word], 10) + 1
        to_be_divided_by += term_frequency[word] ** 2

    to_be_divided_by = math.sqrt(float(to_be_divided_by))

    # Normalize each term frequency value
    for word in term_frequency:
        postings[word][1][postings[word][0].index(doc_id)] = float(term_frequency[word]) / to_be_divided_by


def update_collection_tf(collection_term_frequency, document_term_frequency):
    for word in document_term_frequency:
        if word not in collection_term_frequency:
            collection_term_frequency[word] = 0
        collection_term_frequency[word] += document_term_frequency[word]


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')

    # Initialize out files
    write_dictionary = open(out_dict, "wb")
    write_postings = open(out_postings, "wb")

    # Initialize dictionary and postings
    # dictionary[term][0] = idf
    # dictionary[term][1] = location in postings.txt
    dictionary = {}
    # postings[term][0] = list of doc_ids
    # postings[term][1] = list of tf values
    # postings[term][2][doc_id] = list of positions
    postings = {}

    # Initialize stemmer
    ps = PorterStemmer()

    # Track number of documents
    file_count = 0

    # Initialize term frequency tracker
    collection_term_frequency = {}

    # Nested dictionary to store dataset values
    dataset_dictionary = {}

    # Increase field_size to handle large input of csv
    csv.field_size_limit(sys.maxsize)

    # Read input data from csv files
    with open(os.path.join(in_dir), 'r') as input_csv:
        csv_reader = csv.DictReader(input_csv)
        for line in csv_reader:
            # Initialize term frequency tracker for this document
            document_term_frequency = {}

            # Update file_count
            file_count += 1

            # Read data in dictionary
            doc_id = line[DOC_ID]
            dataset_dictionary[doc_id] = {}
            dataset_dictionary[doc_id][TITLE] = line[TITLE]
            dataset_dictionary[doc_id][CONTENT] = line[CONTENT]
            dataset_dictionary[doc_id][DATE] = line[DATE]
            dataset_dictionary[doc_id][COURT] = line[COURT]

            # Add words from title to postings
            update_postings(doc_id, document_term_frequency, postings, word_tokenize(dataset_dictionary[doc_id][TITLE]), True)

            # Add words from content to postings
            update_postings(doc_id, document_term_frequency, postings, word_tokenize(dataset_dictionary[doc_id][CONTENT]), False)

            # Normalize term frequency for this document and update postings
            normalize_weighted_tf(document_term_frequency, postings, doc_id)

            # Update collection_term_frequency
            update_collection_tf(collection_term_frequency, document_term_frequency)

    # Close csv file
    input_csv.close()

    court_field = generate_court_field(dataset_dictionary)
    date_field = generate_date_field(dataset_dictionary)

    # Convert lists in postings to numpy arrays
    convert_lists_to_nparrays(postings)

    # Set dictionary with idf values and pointers to postings, pickle postings
    for key in sorted(postings):
        dictionary[key] = []
        dictionary[key].append(math.log(file_count / float(collection_term_frequency[key]), 10))
        dictionary[key].append(write_postings.tell())
        pickle.dump(postings[key], write_postings)

    # Pickle dictionary
    pickle.dump(dictionary, write_dictionary)
    pickle.dump(court_field, write_dictionary)
    pickle.dump(date_field, write_dictionary)

    # Close all files
    write_dictionary.close()
    write_postings.close()

    for word in sorted(postings):
        print(word)
        print("postings[word][0]", postings[word][0])
        print("postings[word][1]", postings[word][1])
        for doc in postings[word][2]:
            print("postings[word][2][" + doc + "]", postings[word][2][doc])

    print('indexing completed')


def convert_lists_to_nparrays(postings):
    for word in postings:
        postings[word][0] = np.array(postings[word][0])
        postings[word][1] = np.array(postings[word][1])
        for doc_id in postings[word][2]:
            postings[word][2][doc_id] = np.array(postings[word][2][doc_id])

            # Pickle postings and set pointers in dictionary


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"


if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
