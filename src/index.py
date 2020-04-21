#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import csv
DOC_ID = 1
TITLE = 2
CONTENT = 3
DATE = 4
COURT = 5



def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')

    # nested dictionary to store dataset values
    dataset_dictionary = {}
    print(dataset_dictionary)
    print("\n")

    # read input data from csv files
    with open('testinput.csv', mode='r') as input_csv_file:
        csv_reader = csv.reader(input_csv_file, delimiter=',', quotechar='"')
        line_num = 0
        for line in csv_reader:

            # if not header columns, read into dictionary
            if line_num != 0:
                doc_id = line["document_id"]
                dataset_dictionary[doc_id] = {}
                dataset_dictionary[doc_id][TITLE] = line["title"]
                dataset_dictionary[doc_id][CONTENT] = line["content"]
                dataset_dictionary[doc_id][DATE] = line["date_posted"]
                dataset_dictionary[doc_id][COURT] = line["court"]

    print(dataset_dictionary)
    input_csv_file.close()
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
