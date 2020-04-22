#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import csv
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

    # store court as field
    court_field_dict = {}

    # for each document
    for doc_key in dataset_dict:

        # get court name and make it key of court dict
        court_name = dataset_dict[doc_key][COURT]

        # append doc_id in contents
        if court_name not in court_field_dict:
            court_field_dict[court_name] = list()
        court_field_dict[court_name].append(doc_key)

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

    # store year from title as field
    date_dictionary = {}
    for doc_key in data_dict:

        # get title field
        title = data_dict[doc_key][TITLE]
        year = get_date_from_title(title)
        # append year in contents
        if year not in date_dictionary:
            date_dictionary[year] = list()
        date_dictionary[year].append(doc_key)

    return date_dictionary


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')

    # nested dictionary to store dataset values
    dataset_dictionary = {}

    # increase field_size to handle large input of csv
    csv.field_size_limit(sys.maxsize)

    # read input data from csv files
    with open(in_dir, 'r') as input_csv:
        csv_reader = csv.DictReader(input_csv)
        for line in csv_reader:
            # read data in dictionary
            doc_id = line[DOC_ID]
            dataset_dictionary[doc_id] = {}
            dataset_dictionary[doc_id][TITLE] = line[TITLE]
            dataset_dictionary[doc_id][CONTENT] = line[CONTENT]
            dataset_dictionary[doc_id][DATE] = line[DATE]
            dataset_dictionary[doc_id][COURT] = line[COURT]

    court_field = generate_court_field(dataset_dictionary)
    date_field = generate_date_field(dataset_dictionary)

    # print(dataset_dictionary)
    # print(court_field)
    # print(date_field)

    input_csv.close()
    print('indexing completed')


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
