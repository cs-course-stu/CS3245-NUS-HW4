import os
import re
import sys
import csv
import math
import time
import pickle
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

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
        self.court_field = defaultdict(lambda: [])
        self.date_field = defaultdict(lambda: [])
        self.dictionary = {}
        # dictionary[term] = location in postings.txt

        self.postings = {}
        self.file_handle = None
        # postings[term][0] = idf
        # postings[term][1] = list of doc_ids
        # postings[term][2] = list of tf values
        # postings[term][3] = lists of positions

        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

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

    def update_postings(self, doc_id, doc_terms, words, is_title):
        """
        Method parses words in given sentence and updates postings list
        """
        # Deciding suffix for zones
        if is_title:
            to_append = ".title"
        else:
            to_append = ".content"
            self.average += len(words)

        for i, word in enumerate(words):
            # Pre-process word
            preprocessed_word = self.ps.stem(word)
            current_word = preprocessed_word + to_append

            if current_word in self.postings:
                posting = self.postings[current_word]

                if current_word not in doc_terms:
                    posting[1].append(doc_id)
                    posting[2].append(None)
                    posting[3].append([i])
                else:
                    posting[3][-1].append(i)
            else:
                self.postings[current_word] = [None, [doc_id], [None], [[i]]]

            doc_terms[current_word] += 1

        self.total_doc[doc_id] = len(words)

    def normalize_weighted_tf(self, doc_id, doc_terms):
        """
        normalize weighted term frequency by dividing all term frequency
        with square root(sum of all(square(each term frequency)))
        """
        to_be_divided_by = 0

        # Obtain sum of the square of all term frequency
        for term in doc_terms:
            doc_terms[term] = math.log(doc_terms[term]) + 1
            to_be_divided_by += doc_terms[term] ** 2

        to_be_divided_by = math.sqrt(float(to_be_divided_by))

        # Normalize each term frequency value
        for term in doc_terms:
            self.postings[term][2][-1] = float(doc_terms[term]) / to_be_divided_by

    def convert_lists_to_nparrays(self):
        for key in self.court_field:
            self.court_field[key] = np.array(self.court_field[key])
            np.sort(self.court_field[key])

        self.court_field = dict(self.court_field)

        for key in self.date_field:
            self.date_field[key] = np.array(self.date_field[key])
            np.sort(self.date_field[key])

        self.date_field = dict(self.date_field)

        for word in self.postings:
            self.postings[word][1] = np.array(self.postings[word][1])
            self.postings[word][2] = np.array(self.postings[word][2])
            for i in range(len(self.postings[word][3])):
                self.postings[word][3][i] = np.array(self.postings[word][3][i])

    def update_idf(self):
        self.file_count = len(self.total_doc)

        for word in self.postings:
            df = len(self.postings[word][1])
            idf = math.log(self.file_count / df, 10)
            self.postings[word][0] = idf

    def remove_punctuation(self, terms):

        modified_term = re.sub(r"[,;@#?!&$()%°~^_.+=\"><`|}{*:/]+ *", " ", terms)

        return modified_term

    def tokenize_and_remove_stopwords(self, input):
        token_words = word_tokenize(input)
        lower_token_words = []
        for words in token_words:
            lower_token_words.append(words.lower())

        filtered_words = []

        for word in lower_token_words:
            if word not in self.stop_words:
                filtered_words.append(word)

        return filtered_words

    def build_index(self, in_dir):
        """
        build index from documents stored in the input directory,
        then output the dictionary file and postings file
        """
        print('indexing...')

        # Increase field_size to handle large input of csv
        max_int = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.

            try:
                csv.field_size_limit(max_int)
                break
            except OverflowError:
                max_int = int(max_int / 10)

        total_count = 0
        repeated_file_count = 0

        # Read input data from csv files
        with open(os.path.join(in_dir), 'r', encoding="utf8") as input_csv:
            csv_reader = csv.DictReader(input_csv)
            for line in csv_reader:
                total_count += 1
                if total_count % 100 == 0:
                    print(total_count)

                if line is None:
                    continue
                if total_count >= 3:
                    break

                # Determine whether the document has been processed
                doc_id = line[DOC_ID]
                if doc_id in self.total_doc:
                    repeated_file_count += 1
                    continue

                # remove punctuation for each column
                line[TITLE] = self.remove_punctuation(line[TITLE])
                line[CONTENT] = self.remove_punctuation(line[CONTENT])
                line[DATE] = self.remove_punctuation(line[DATE])
                line[COURT] = self.remove_punctuation(line[COURT])

                # build index for one doc
                doc_terms = defaultdict(lambda: 0)

                # Add words from title to postings
                tokens = self.tokenize_and_remove_stopwords(line[TITLE])
                self.update_postings(doc_id, doc_terms, tokens, True)

                # Add words from content to postings
                tokens = self.tokenize_and_remove_stopwords(line[CONTENT])
                self.update_postings(doc_id, doc_terms, tokens, False)

                # Normalize term frequency for this document and update postings
                self.normalize_weighted_tf(doc_id, doc_terms)

                # Append court and date fields
                line[DATE] = self.get_date_from_title(line[TITLE])
                self.date_field[line[DATE]].append(doc_id)
                self.court_field[line[COURT]].append(doc_id)

                # print("read doc from csv: ")
                # print(file_count)

        # Set idf values in postings
        self.update_idf()

        # Convert lists in postings and fields to numpy arrays
        self.convert_lists_to_nparrays()

        # Calculate average
        self.average /= len(self.total_doc)

        # for word in sorted(postings):
        #     print(word)
        #     print("postings[word][0]", postings[word][0])
        #     print("postings[word][1]", postings[word][1])
        #     for doc in postings[word][2]:
        #         print("postings[word][2][" + doc + "]", postings[word][2][doc])

        print("____________")
        print(total_count)
        print(repeated_file_count)
        print(self.file_count)
        print("____________")

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
            # pickle.dump(self.postings[key][0], write_postings)
            # pickle.dump(self.postings[key][1], write_postings)
            # pickle.dump(self.postings[key][2], write_postings)
            # pickle.dump(self.postings[key][3], write_postings)
            # np.save(write_postings, self.postings[key][0])
            # np.save(write_postings, self.postings[key][1])
            # np.save(write_postings, self.postings[key][2])
            # np.save(write_postings, self.postings[key][3])
            # for i in range(len(self.postings[key][3])):
            #     np.save(write_postings, self.postings[key][3][i])

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
        # print('idf:', indexer.postings[key][0])
        # print('doc:', indexer.postings[key][1])
        # print('tfs:', indexer.postings[key][2])
        # print('position:', indexer.postings[key][3])
        if not self.file_handle:
            self.file_handle = open(self.postings_file, 'rb')
        
        rst = {}
        for term in terms:
            if term in self.dictionary:
                self.file_handle.seek(self.dictionary[term])
                # load postings and position
                info = np.load(self.file_handle, allow_pickle=True)
                rst[term] = tuple(info)
                # log_tf = np.load(self.file_handle, allow_pickle=True)

                # # load position
                # if(self.phrasal_query):
                #     position = np.load(self.file_handle, allow_pickle=True)
                #     ret[term] = (doc, log_tf, position)
                # else:
                #     ret[term] = (doc, log_tf, )

            else:
                idf = np.empty(shape=(0, ), dtype=np.float32)
                doc = np.empty(shape=(0, ), dtype=np.int32)
                tfs = np.empty(shape=(0, ), dtype=np.float32)
                position = np.empty(shape=(0, ), dtype=object)
                # if self.phrasal_query:

                #     ret[term] = (doc, log_tf, position)
                # else:
                #     ret[term] = (doc, log_tf, )
                rst[term] = (idf, doc, tfs, position)

        return rst

if __name__ == '__main__':
    indexer = Indexer('dictionary.txt', 'postings.txt')

    # start = time.time()
    # indexer.build_index('/Users/wangyifan/Google Drive/hw_4/dataset.csv')
    # mid = time.time()
    # indexer.SavetoFile()
    # end = time.time()

    # print('build time: ' + str(mid - start))
    # print('dump time: ' + str(end - mid))
    average, total_doc, court_field, date_field, dictionary = indexer.LoadDict()
    # print(dictionary)
    terms = ['\'s.content','123']
    print(indexer.LoadTerms(terms))
    # for key in indexer.postings:
    #     print('key:', key)
    #     print('idf:', indexer.postings[key][0])
    #     print('doc:', indexer.postings[key][1])
    #     print('tfs:', indexer.postings[key][2])
    #     print('position:', indexer.postings[key][3])
