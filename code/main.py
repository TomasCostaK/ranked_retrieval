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

        # used in bm25 to check each documents length, and the average of all docs
        self.docs_length = {}

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

                        self.docs_length[index] = len(tokens)
                        self.collection_size += 1 
            
                #print("Estimated tokenizing/stemming time: %.4fs" % (toc-tic)) #useful for debugging

                self.indexer.index(tokens, index)
                #print("Estimated indexing time: %.4fs" % (toc-tic)) #useful for debugging

        self.indexed_map = self.indexer.getIndexed()


    def process_queries(self, path='../content/queries.txt', mode='tf_idf', k1=1.2, b=0.75 ,docs_limit=10):
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
                    print("\t Documents retrieved in %.0f ms" % ((toc-tic) *1000))




        elif mode=='bm25':
            #Show results for ranking
            with open(path,'r') as f:
                query_n = 1
                self.queries_results()
                for query in f.readlines():
                    #print("Results for query: %s\n" % (query))
                    tic = time.time()
                    best_docs = self.rank_bm25(query, k1, b, docs_limit)
                    #for doc in best_docs:
                        #print("Document: %s \t with score: %.5f" % (doc[0], doc[1]))
                    toc = time.time()
                    #print("\t Documents retrieved in %.4f ms" % ((toc-tic) *1000))

                    docs_ids = [doc_id for doc_id, score in best_docs]
                    
                    # evaluate each query and print a table
                    self.evaluate_query(query_n, docs_ids)
                    query_n += 1
        else:
            usage()
            sys.exit(1)

    def queries_results(self):
        print("\tPrecision \tRecall	\tF-measure	\tAverage \tPrecision \tNDCG \tLatency\nQuery #	@10	@20	@50	@10	@20	@50	@10	@20	@50	@10	@20	@50	@10	@20	@50")

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
        
        # TODO, this normalization is wrong, see where i can add the doc_length
        length_normalize = math.sqrt(sum([x**2 for x in query_weights_list])) * math.sqrt(sum([x**2 for x in documents_weights_list]))
            
        #find a better way to normalize data
        for k, v in best_docs.items():
            best_docs[k] = v/length_normalize
        
        most_relevant_docs = sorted(best_docs.items(), key=lambda x: x[1], reverse=True)
        return most_relevant_docs[:docs_limit]


    def rank_bm25(self, query, k1, b, docs_limit=10):
        # declaration of vars to be used in tf.idf
        best_docs = collections.defaultdict(lambda: 0) # default start at 0 so we can do cumulative gains
        N = self.collection_size

        # Special call to indexer, so we can access the term frequency, making use of modularization
        indexed_query = self.indexer.index_query(self.tokenizer.tokenize(query,-1))

        for term,tf_query in indexed_query.items():
            #special treatment, weights at 0
            if term not in self.indexed_map.keys():
                continue

            df = self.indexed_map[term]['doc_freq']

            # calculate idf for each term
            # TODO, ask teacher if this IDF is calculated well
            idf = math.log10(self.collection_size / self.indexed_map[term]['doc_freq'])
            self.indexed_map[term]['idf'] = idf

            avdl = sum([ value for key,value in self.docs_length.items()]) / self.collection_size
            # now we iterate over every term
            for doc_id, tf_doc in self.indexed_map[term]['doc_ids'].items():
                dl = self.docs_length[doc_id]
                score = self.calculate_BM25(df, k1, b, dl, avdl, tf_doc)
                best_docs[doc_id] += score
        
        most_relevant_docs = sorted(best_docs.items(), key=lambda x: x[1], reverse=True)
        return most_relevant_docs[:docs_limit]

    def calculate_BM25(self, df, k1, b, dl, avdl, tf_doc):
        N = self.collection_size

        #TODO, confirm this is correct
        term1 = math.log(N/df)
        term2 = ((k1 + 1) * tf_doc) / ( k1 * ((1-b) + b*dl/avdl) + tf_doc )
        return term1*term2


    def evaluate_query(self, query_n, docs_ids):
        #initiate counts at 0
        fp = 0
        tp = 0
        fn = 0

        #Open queries relevance
        with open('../content/queries.relevance.filtered.txt','r') as q_f:
            for q_relevance in q_f.readlines():
                query_relevance_array = q_relevance.split(" ") # 1st is query number, 2nd is document id, 3rd is relevance
                
                if int(query_relevance_array[0]) == query_n:
                    # if relevant and not showing up - FN
                    if int(query_relevance_array[2]) > 0 and query_relevance_array[1] not in docs_ids:
                        fn += 1

                    # if showing up but not relevant - FP
                    if int(query_relevance_array[2]) == 0 and query_relevance_array[1] in docs_ids:
                        fp += 1

                    # if showing up and relevant - TP
                    if int(query_relevance_array[2]) > 0 and query_relevance_array[1] in docs_ids:
                        tp += 1                    
                
                elif int(query_relevance_array[0]) > query_n:
                    continue
        
            # returned values
            # TODO, are the special cases necessary?
            try:
                precision = tp / (fp + tp)
            except ZeroDivisionError:
                precision = 0
            
            try:
                recall = tp / ( tp + fn)
            except ZeroDivisionError:
                recall = 0
                
            if recall + precision == 0:
                f_score = 0
            else:
                f_score = (2 * recall * precision) / (recall + precision)

            
            print("Query: %d  %.3f \t %.3f \t \t %.3f" % (query_n,precision,recall, f_score))

        return precision, recall, f_score

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
    k1 = 1.2
    b = 0.75
    rtli.process_queries('../content/queries.txt',k1=k1, b=b,mode='bm25', docs_limit=10)
    #rtli.process_queries('queries.txt',mode='tf_idf')
    toc = time.time()
    print("Time spent ranking documents: %.4fs" % (toc-tic))

    rtli.write_index_file()