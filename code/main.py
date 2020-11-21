from tokenizer import Tokenizer
from indexer import Indexer
import time
import sys
import operator
import os
import csv
import math
import collections

"""
Authors:
Tomás Costa - 89016  
André Gual - 88751
"""


class RTLI:  # Reader, tokenizer, linguistic, indexer
    def __init__(self, tokenizer_mode, file='../content/metadata_2020-03-27.csv', stopwords_file="../content/snowball_stopwords_EN.txt", chunksize=10000):
        self.tokenizer = Tokenizer(tokenizer_mode, stopwords_file)
        self.indexer = Indexer()
        self.file = file

        # defines the number of lines to be read at once
        self.chunksize = chunksize

        # tryout for new structure in dict
        self.indexed_map = {}

        # collection size
        self.collection_size = 0


    def gen_chunks(self, reader):
        chunk = []
        for i, line in enumerate(reader):
            if (i % self.chunksize == 0 and i > 0):
                yield chunk
                del chunk[:]  # or: chunk = []
            chunk.append(line)
        yield chunk

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
                    self.collection_size += 1 
            
                #print("Estimated tokenizing/stemming time: %.4fs" % (toc-tic)) #useful for debugging

                self.indexer.index(tokens, index)
                #print("Estimated indexing time: %.4fs" % (toc-tic)) #useful for debugging

        self.indexed_map = self.indexer.getIndexed()


    def process_queries(self, path='../content/queries.txt', mode='tf_idf',docs_limit=10):
        if mode=='tf_idf':
            #Show results for ranking
            with open(path,'r') as f:
                for query in f.readlines():
                    print("\nResults for query: %s\n" % (query))
                    tic = time.time()
                    best_docs = self.rank_tf_idf(query,docs_limit)
                    for doc in best_docs:
                        print("Document: %s \t with relevance: %.5f" % (doc[0], doc[1]))
                    toc = time.time()
                    print("\t Documents retrieved in %.4f seconds" % (toc-tic))


        elif mode=='bm25':
            #Show results for ranking
            with open(path,'r') as f:
                for query in f.readlines:
                    print("Results for query: %s\n" % (query))
                    best_docs = self.rank_bm25(query,docs_limit)
                    print("Document: %s \t with relevance: %.5f" % (doc[0], doc[1]))
        else:
            usage()
            sys.exit(1)


    def rank_tf_idf(self, query, docs_limit=10):
        # declaration of vars to be used in tf.idf
        best_docs = collections.defaultdict(lambda: 0) # default start at 0 so we can do cumulative gains
        N = self.collection_size

        # Special call to indexer, so we can access the term frequency, making use of modularization
        indexed_query = self.indexer.index_query(self.tokenizer.tokenize(query,-1))
        query_weights_list = []
        documents_weights_list = []

        for term,tf_query in indexed_query.items():
            #special treatment, weights at 0
            if term not in self.indexed_map.keys():
                continue
            
            tf_weight = math.log10(tf_query) + 1
            df = self.indexed_map[term]['doc_freq']
            idf = math.log10(N/float(df))

            weight_query_term = tf_weight * idf #this is the weight for the term in the query
            query_weights_list.append(weight_query_term)

            # now we iterate over every term
            for doc_id, tf_doc in self.indexed_map[term]['doc_ids'].items():
                tf_doc_weight = math.log10(tf_doc) + 1
                documents_weights_list.append(tf_doc_weight)
                
                """ #TODO these calculus are wrong in the calculator, talk to prof
                documents_weights_list = [0,1.3,2,3]
                query_weights_list = [1,0,1,1.3]
                #normalization step
                """
                best_docs[doc_id] += (weight_query_term * tf_doc_weight) 
        
        length_normalize = math.sqrt(sum([x**2 for x in query_weights_list])) * math.sqrt(sum([x**2 for x in documents_weights_list]))
            
        #find a better way to normalize data
        #best_docs /= length_normalize #now we normalize the results, and reset the arrays
        for k, v in best_docs.items():
            best_docs[k] = v/length_normalize
        
        most_relevant_docs = sorted(best_docs.items(), key=lambda x: x[1], reverse=True)
        return most_relevant_docs[:docs_limit]
        """
            for key,value in self.indexed_map.items():
                term_dict = self.indexed_map[key]['doc_ids']
                
                # iterate every doc_id in term and associate its tdf
                for doc_id,count in value['doc_ids'].items():
                    term_dict[doc_id] = math.log10(count)+1
                    self.indexed_map[key]['doc_ids'] = term_dict

                # calculate idf for each term
                idf = math.log10(self.collection_size / value['doc_freq'])
                value['idf'] = idf
            """
        #print("Indexed map: ", self.indexed_map)

    def rank_bm25(self, query, docs_limit=10):
        return 0

    def write_index_file(self, file_output='../output/indexed_map.txt'):

        with open(file_output,'w+') as f:
            for term, value in self.indexed_map.items(): 
                string = term + ": " + str(value)
                f.write(string)


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
    print("Usage: python3 main.py <complex/simple> <chunksize>")

if __name__ == "__main__":  # maybe option -t simple or -t complex

    if len(sys.argv) < 3:
        usage()
        sys.exit(1)

    if sys.argv[1] == "complex":
        rtli = RTLI(tokenizer_mode="complex",chunksize=int(sys.argv[2]))

    elif sys.argv[1] == "simple":
        rtli = RTLI(tokenizer_mode="simple",chunksize=int(sys.argv[2]))

    else:
        print("Usage: python3 main.py <complex/simple> <chunksize>")
        sys.exit(1)

    tic = time.time()
    rtli.process()
    toc = time.time()

    rtli.domain_questions(toc-tic)
    
    tic = time.time()
    rtli.process_queries('../content/queries.txt',mode='tf_idf')
    #rtli.process_queries('queries.txt',mode='bm25')
    toc = time.time()
    print("Time spent ranking documents: %.4fs" % (toc-tic))

    rtli.write_index_file()