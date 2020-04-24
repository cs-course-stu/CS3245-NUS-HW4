import csv
import numpy as np
import os
import sys
import pickle
import math
import time
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

DOC_ID = "document_id"
TITLE = "title"
CONTENT = "content"
DATE = "date_posted"
COURT = "court"


class Indexer:
    """ class Indexer is a class dealing with building index, saving it to file and loading it
    Args:
        dictionary_file: the name of the dictionary.
        postings_file: the name of the postings
    """

    def __init__(self, dictionary_file, postings_file):
        self.dictionary_file = dictionary_file
        self.postings_file = postings_file
        self.average = 0
        self.total_doc = {}
        self.file_count = 0
        self.collection_term_frequency = {}
        self.court_field = {}
        self.date_field = {}
        # dictionary[term][0] = idf
        # dictionary[term][1] = location in postings.txt
        self.dictionary = {}
        # postings[term][0] = list of doc_ids
        # postings[term][1] = list of tf values
        # postings[term][2][doc_id] = list of positions
        self.postings = {}

    def generate_court_field(self, dataset_dict):
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

    def get_date_from_title(self, title):
        """
        Method to extract year from title
        """
        if title is None:
            return -1

        terms = title.split()

        if terms is None:
            return -1

        if len(terms) < 3:
            return -1

        brac_year = terms[-3]
        list_year = []
        for char in brac_year:
            if char != "[" and char != "]":
                list_year.append(char)

        year = "".join(list_year)
        return year

    def generate_date_field(self, data_dict):
        """
        Method to generate the postings list of documents for the court field. key is
        court name and value is the list of documents that are in court.
        """

        # Store year from title as field
        date_dictionary = {}
        for doc_key in data_dict:

            # Get title field
            title = data_dict[doc_key][TITLE]
            year = self.get_date_from_title(title)
            # Append year in contents
            if year not in date_dictionary:
                date_dictionary[year] = list()
            date_dictionary[year].append(doc_key)

        # sort doc ids in value
        for key in date_dictionary:
            date_dictionary[key] = sorted(date_dictionary[key])

        return date_dictionary

    def update_postings(self, doc_id, document_term_frequency, words, is_title):
        """
        Method parses words in given sentence and updates postings list
        """
        # Deciding suffix for zones
        if is_title:
            to_append = ".title"
        else:
            to_append = ".content"

        # Initialise PorterStemmer
        ps = PorterStemmer()

        for i in range(len(words)):
            # Pre-process word
            preprocessed_word = ps.stem(words[i]).lower()
            current_word = preprocessed_word + to_append
            # current_word = words[i].append(toAppend)

            if current_word not in self.postings:
                self.postings[current_word] = []
                self.postings[current_word].append(None)
                self.postings[current_word].append([])
                self.postings[current_word].append([])
                self.postings[current_word].append({})
            if doc_id not in self.postings[current_word][1]:
                self.postings[current_word][1].append(doc_id)
                self.postings[current_word][2].append(None)
                self.postings[current_word][3][doc_id] = []
            self.postings[current_word][3][doc_id].append(i)
            if current_word not in document_term_frequency:
                document_term_frequency[current_word] = 0
            document_term_frequency[current_word] += 1

    def normalize_weighted_tf(self, term_frequency, doc_id):
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
        self.total_doc[doc_id] = to_be_divided_by

        # Normalize each term frequency value
        for word in term_frequency:
            self.postings[word][2][self.postings[word][1].index(doc_id)] = float(term_frequency[word]) / to_be_divided_by

    def update_collection_tf(self, document_term_frequency):
        for word in document_term_frequency:
            if word not in self.collection_term_frequency:
                self.collection_term_frequency[word] = 0
            self.collection_term_frequency[word] += document_term_frequency[word]

    def convert_lists_to_nparrays(self):
        for word in self.postings:
            self.postings[word][1] = np.array(self.postings[word][1])
            self.postings[word][2] = np.array(self.postings[word][2])
            for doc_id in self.postings[word][3]:
                self.postings[word][3][doc_id] = np.array(self.postings[word][3][doc_id])

    def update_idf(self):
        for word in self.postings:
            self.postings[word][0] = math.log(self.file_count / float(self.collection_term_frequency[word]), 10)

    def build_index(self, in_dir):
        """
        build index from documents stored in the input directory,
        then output the dictionary file and postings file
        """
        print('indexing...')

        # Initialize stemmer
        ps = PorterStemmer()

        # Nested dictionary to store dataset values
        dataset_dictionary = {}

        # Increase field_size to handle large input of csv
        csv.field_size_limit(sys.maxsize)
        # max_int = sys.maxsize
        # while True:
        #     # decrease the maxInt value by factor 10
        #     # as long as the OverflowError occurs.
        #
        #     try:
        #         csv.field_size_limit(max_int)
        #         break
        #     except OverflowError:
        #         max_int = int(max_int / 10)

        # Read input data from csv files
        with open(os.path.join(in_dir), 'r', encoding="utf8") as input_csv:
            csv_reader = csv.DictReader(input_csv)
            for line in csv_reader:

                if line is None:
                    continue

                # Read data in dictionary
                doc_id = line[DOC_ID]
                dataset_dictionary[doc_id] = {}
                dataset_dictionary[doc_id][TITLE] = line[TITLE]
                dataset_dictionary[doc_id][CONTENT] = line[CONTENT]
                dataset_dictionary[doc_id][DATE] = line[DATE]
                dataset_dictionary[doc_id][COURT] = line[COURT]

                # print("read doc from csv: ")
                # print(file_count)

        # Close csv file
        input_csv.close()

        # Start indexing
        for doc_id in dataset_dictionary:

            # Update file_count
            self.file_count += 1
            print(self.file_count)

            # Initialize term frequency tracker for this document
            document_term_frequency = {}

            # Add words from title to postings
            self.update_postings(doc_id, document_term_frequency, word_tokenize(dataset_dictionary[doc_id][TITLE]), True)

            # Add words from content to postings
            self.update_postings(doc_id, document_term_frequency, word_tokenize(dataset_dictionary[doc_id][CONTENT]), False)

            # print("processed doc from csv: ")
            # print(file_count)
            # print("______________________________________")

            # Normalize term frequency for this document and update postings
            self.normalize_weighted_tf(document_term_frequency, doc_id)

            # Update collection_term_frequency
            self.update_collection_tf(document_term_frequency)

        # Set idf values in postings
        self.update_idf()

        # Generate court and date fields
        self.court_field = self.generate_court_field(dataset_dictionary)
        self.date_field = self.generate_date_field(dataset_dictionary)

        # Convert lists in postings to numpy arrays
        self.convert_lists_to_nparrays()

        # Calculate average
        self.average = (self.file_count * (self.file_count + 1) / 2) / self.file_count

        # for word in sorted(postings):
        #     print(word)
        #     print("postings[word][0]", postings[word][0])
        #     print("postings[word][1]", postings[word][1])
        #     for doc in postings[word][2]:
        #         print("postings[word][2][" + doc + "]", postings[word][2][doc])

        print('indexing completed')

    def SavetoFile(self):
        """
        save dictionary, postings and skip pointers given fom build_index() to file
        """

        print('saving to file...')

        # Initialize out files
        write_dictionary = open(self.dictionary_file, "wb")
        write_postings = open(self.postings_file, "wb")

        # Set dictionary with idf values and pointers to postings, pickle postings
        for key in sorted(self.postings):
            self.dictionary[key] = write_postings.tell()
            pickle.dump(self.postings[key], write_postings)

        # Pickle dictionary
        pickle.dump(self.average, write_dictionary)
        pickle.dump(self.total_doc, write_dictionary)
        pickle.dump(self.court_field, write_dictionary)
        pickle.dump(self.date_field, write_dictionary)
        pickle.dump(self.dictionary, write_dictionary)

        # Close all files
        write_dictionary.close()
        write_postings.close()

        print('save to file successfully!')

    def LoadDict(self):
        """ load dictionary from file
        Returns:
            total_doc: total doc_id
            dictionary: all word list
        """
        print('loading dictionary...')

        with open(self.dictionary_file, 'rb') as f:
            self.average = pickle.load(f)
            self.total_doc = pickle.load(f)
            self.court_field = pickle.load(f)
            self.date_field = pickle.load(f)
            self.dictionary = pickle.load(f)

        print('load dictionary successfully!')
        return self.average, self.total_doc, self.court_field, self.date_field, self.dictionary

    def LoadTerms(self, terms):
        """ load multiple postings lists from file
        Args:
            terms: the list of terms need to be loaded
        Returns:
            postings_lists: the postings lists correspond to the terms
        """
        postings_lists = {}

        return postings_lists
