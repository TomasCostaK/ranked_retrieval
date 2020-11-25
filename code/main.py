from tokenizer import Tokenizer
from indexer import Indexer
from ranker import Ranker
import time

from numpy import cumsum
import sys
import operator
import os
import csv
import math
import collections

"""
Authors:
Tomás Costa - 89016  
"""


class RTLI:  # Reader, tokenizer, linguistic, indexer
    def __init__(self, tokenizer_mode, file='../content/metadata_2020-03-27.csv', stopwords_file="../content/snowball_stopwords_EN.txt", chunksize=10000, queries_path='../content/queries.txt' ,rank_mode='bm25', docs_limit=50):
        self.tokenizer = Tokenizer(tokenizer_mode, stopwords_file)
        self.indexer = Indexer()
        self.ranker = Ranker(queries_path=queries_path ,mode=rank_mode,docs_limit=docs_limit)
        self.file = file

        # defines the number of lines to be read at once
        self.chunksize = chunksize

        # tryout for new structure in dict
        self.indexed_map = {}

        # used in bm25 to check each documents length, and the average of all docs
        self.docs_length = {}

        # collection size
        self.collection_size = 0

    # auxiliary function to generate chunks of text to read
    def gen_chunks(self, reader):
        chunk = []
        for i, line in enumerate(reader):
            if (i % self.chunksize == 0 and i > 0):
                yield chunk
                del chunk[:]  # or: chunk = []
            chunk.append(line)
        yield chunk

    # main function of indexing and tokenizing
    def process(self):
        tokens = []

        # Reading step
        # We passed the reader to here, so we could do reading chunk by chunk
        with open(self.file, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for chunk in self.gen_chunks(reader):
                for row in chunk:
                    index = row['cord_uid']
                    # Tokenizer step
                    if row['abstract'] != "":
                        appended_string = row['abstract'] + " " + row['title']
                        tokens += self.tokenizer.tokenize(appended_string, index)

                        self.docs_length[index] = len(tokens)
                        self.collection_size += 1 
            
                #print("Estimated tokenizing/stemming time: %.4fs" % (toc-tic)) #useful for debugging

                self.indexer.index(tokens, index)
                #print("Estimated indexing time: %.4fs" % (toc-tic)) #useful for debugging

        self.indexed_map = self.indexer.getIndexed()
        self.updateIdfs()

    def rank(self, analyze_table, tokenizer_mode):
        self.ranker.update(self.docs_length, self.collection_size, self.indexed_map, tokenizer_mode, "../content/snowball_stopwords_EN.txt")
        self.ranker.process_queries(analyze_table=analyze_table)

    # we call this extra step, so every term has an idf
    def updateIdfs(self):
        for term, value in self.indexed_map.items():
            idf = math.log10(self.collection_size / self.indexed_map[term]['doc_freq'])
            self.indexed_map[term]['idf'] = idf


    # function to write indexed terms to file, in a similar output to the one requested
    def write_index_file(self, file_output='../output/indexed_map.txt'):
        with open(file_output,'w+') as f:
            for term, value in self.indexed_map.items(): 
                string = term + ": " + str(value) + '\n'
                f.write(string)


    # Questions being asked in work nº1
    def domain_questions(self, time):
        # Question a)
        mem_size = self.calculate_dict_size(self.indexed_map) / 1024 / 1024
        print("A) Estimated process time: %.4fs and spent %.2f Mb of memory" %
            (time, mem_size))

        # Question b)
        vocab_size = len(self.indexed_map.keys())
        print("B) Vocabulary size is: %d" % (vocab_size))

        # Question c)
        # i think we can do this, because these keys only have 1 value, which is the least possible to get inserted into the dict
        ten_least_frequent = [key for (key, value) in sorted(
            self.indexed_map.items(), key=lambda x: x[1]['col_freq'], reverse=False)[:10]]
        # sort alphabetical
        #ten_least_frequent.sort()
        print("\nC) Ten least frequent terms:")
        for term in ten_least_frequent:
            print(term)

        # Question d)
        # i think we can do this, because these keys only have 1 value, which is the least possible to get inserted into the dict
        ten_most_frequent = [key for (key, value) in sorted(
            self.indexed_map.items(), key=lambda x: x[1]['col_freq'], reverse=True)[:10]]
        # sort alphabetical
        #ten_most_frequent.sort()
        print("\nD) Ten most frequent terms:")
        for term in ten_most_frequent:
            print(term)

    # auxiliary function to calculate dict size recursively
    def calculate_dict_size(self, input_dict):
        mem_size = 0
        for key, value in input_dict.items():
            # in python they dont count size, so we have to do it iteratively
            mem_size += sys.getsizeof(value)
            for key2, value2 in value['doc_ids'].items(): 
                mem_size += sys.getsizeof(value2)

        # adding the own dictionary size
        return mem_size + sys.getsizeof(input_dict)

def usage():
    print("Usage: python3 main.py <tokenizer_mode: complex/simple> <chunksize:int> <ranking_mode:tf_idf/bm25> <analyze_table:boolean>")

if __name__ == "__main__":  

    # work nº2 defaults
    mode = 'tf_idf'
    analyze_table = True
    docs_limit = 20
    tokenizer_mode = 'complex'

    if len(sys.argv) < 3:
        usage()
        sys.exit(1)

    if sys.argv[1] == "complex":
        rtli = RTLI(tokenizer_mode="complex",chunksize=int(sys.argv[2]), rank_mode=mode, docs_limit=docs_limit)

    elif sys.argv[1] == "simple":
        rtli = RTLI(tokenizer_mode="simple",chunksize=int(sys.argv[2]), rank_mode=mode, docs_limit=docs_limit)

    else:
        print("Usage: python3 main.py <complex/simple> <chunksize>")
        sys.exit(1)

    tic = time.time()
    rtli.process()
    toc = time.time()

    rtli.domain_questions(toc-tic)


    tic = time.time()
    rtli.rank(analyze_table, tokenizer_mode)
    toc = time.time()

    rtli.write_index_file()