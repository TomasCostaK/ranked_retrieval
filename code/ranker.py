from numpy import cumsum
from indexer import Indexer
from tokenizer import Tokenizer

import sys
import time
import math
import collections

class Ranker:
    def __init__(self, queries_path='../content/queries.txt', k1=1.2, b=0.75, mode='tf_idf', docs_limit=50, docs_length={}):
        #values used for bm25, these are the most used defaults
        self.k1 = k1
        self.b = b

        self.indexer = Indexer()
        
        #file location for the queries
        self.queries_path = queries_path

        #type of ranking mode
        self.mode = mode
        
        #limit of docs being analyzed, usually 50 for seeing the table
        self.docs_limit = docs_limit

        # used in ranking to check each documents length, and the average of all docs
        self.docs_length = docs_length

        # arrays used to calculate means
        self.mean_precision_array = []
        self.mean_recall_array = []
        self.mean_f_measure_array = []
        self.mean_ap_array = []
        self.mean_ndcg_array = []
        self.mean_latency_array = []


    def update(self, docs_len, collection_size, indexed_words, tokenizer_mode, stopwords_file):
        self.docs_length = docs_len
        # atributes used in calculus
        self.indexed_map = indexed_words
        self.collection_size = collection_size
        self.tokenizer = Tokenizer(tokenizer_mode, stopwords_file)
   
    def process_queries(self, analyze_table=True):
        #Show results for ranking
        tic_total = time.time()
        with open(self.queries_path,'r') as f:
            query_n = 1
            if analyze_table:
                self.queries_results()
            for query in f.readlines():
                tic = time.time()
                
                if self.mode == 'tf_idf':
                    best_docs = self.rank_tf_idf(query)
                elif self.mode == 'bm25':
                    best_docs = self.rank_bm25(query)
                else:
                    usage()
                    sys.exit(1)

                if not analyze_table:
                    print("Results for query: %s\n" % (query))
                    for doc in best_docs:
                        print("Document: %s \t with score: %.5f" % (doc[0], doc[1]))
                else:
                    docs_ids = [doc_id for doc_id, score in best_docs]
                    # evaluate each query and print a table
                    toc = time.time()
                    self.evaluate_query(query_n, docs_ids, toc-tic)
                
                # update query number
                query_n += 1
        
        toc_total = time.time()
        # calculate means, we do it like this, so its easier to read
        mean_precision = sum(self.mean_precision_array) / len(self.mean_precision_array)
        mean_recall = sum(self.mean_recall_array) / len(self.mean_recall_array)
        mean_f_measure = sum(self.mean_f_measure_array) / len(self.mean_f_measure_array)
        mean_ap = sum(self.mean_ap_array) / len(self.mean_ap_array)
        mean_ndcg = sum(self.mean_ndcg_array) / len(self.mean_ndcg_array)
        mean_latency = sum(self.mean_latency_array) / len(self.mean_latency_array)


        print("Median: \t %.3f \t\t\t %.3f \t\t\t  %.3f \t\t  %.3f \t\t  %.3f \t\t  %.0fms " % \
            (mean_precision, mean_recall, mean_f_measure, mean_ap, mean_ndcg, mean_latency*1000)
        )
        print("Query throughput: %.3f queries per second" % ( 1 * 50 / (toc_total-tic_total) ))

    def queries_results(self):
        print("  \t\tPrecision \t\t Recall  	\tF-measure     \tAverage Precision \tNDCG \t\t\t Latency\nQuery #	@10	@20	@50	@10	@20	@50	@10	@20	@50	@10	@20	@50	@10	@20	@50")

    def rank_tf_idf(self, query):
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
            idf = self.indexed_map[term]['idf']

            weight_query_term = tf_weight * idf #this is the weight for the term in the query
            query_weights_list.append(weight_query_term)

            # now we iterate over every term
            for doc_id, tf_doc in self.indexed_map[term]['doc_ids'].items():
                tf_doc_weight = math.log10(tf_doc) + 1
                documents_weights_list.append(tf_doc_weight)
                best_docs[doc_id] += (weight_query_term * tf_doc_weight) 
        
        # TODO, this normalization is wrong, see where i can add the doc_length
        length_normalize = math.sqrt(sum([x**2 for x in query_weights_list])) * math.sqrt(sum([x**2 for x in documents_weights_list]))
            
        #find a better way to normalize data
        for k, v in best_docs.items():
            best_docs[k] = v/length_normalize
        
        most_relevant_docs = sorted(best_docs.items(), key=lambda x: x[1], reverse=True)
        return most_relevant_docs[:self.docs_limit]


    def rank_bm25(self, query):
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
            idf = self.indexed_map[term]['idf']

            avdl = sum([ value for key,value in self.docs_length.items()]) / self.collection_size
            # now we iterate over every term
            for doc_id, tf_doc in self.indexed_map[term]['doc_ids'].items():
                dl = self.docs_length[doc_id]
                score = self.calculate_BM25(df, dl, avdl, tf_doc)
                best_docs[doc_id] += score
        
        most_relevant_docs = sorted(best_docs.items(), key=lambda x: x[1], reverse=True)
        return most_relevant_docs[:self.docs_limit]

    # auxiliary function to calculate bm25 formula
    def calculate_BM25(self, df, dl, avdl, tf_doc):
        N = self.collection_size
        term1 = math.log(N/df)
        term2 = ((self.k1 + 1) * tf_doc) / ( self.k1 * ((1-self.b) + self.b*dl/avdl) + tf_doc )
        return term1*term2

    def evaluate_query(self, query_n, docs_ids, latency):
        #initiate counts at 0
        fp = 0
        tp = 0
        fn = 0

        for i in range(0,3):

            if i==0:
                docs_ids_new = docs_ids[:10]
            elif i==1:
                docs_ids_new = docs_ids[:20]
            elif i==2:
                docs_ids_new = docs_ids[:50]

            #Open queries relevance
            with open('../content/queries.relevance.filtered.txt','r') as q_f:
                # variables for average precision
                doc_counter = 0
                docs_ap = []

                # variables for ndcg
                relevance_ndcg = []

                for q_relevance in q_f.readlines():
                    
                    query_relevance_array = q_relevance.split(" ") # 1st is query number, 2nd is document id, 3rd is relevance
                    
                    if int(query_relevance_array[0]) == query_n:

                        # if relevant and not showing up - FN
                        if int(query_relevance_array[2]) > 0 and query_relevance_array[1] not in docs_ids_new:
                            fn += 1

                        # if showing up but not relevant - FP
                        if int(query_relevance_array[2]) == 0 and query_relevance_array[1] in docs_ids_new:
                            fp += 1
                            # treatment for ndcg
                            relevance_ndcg.append(float(query_relevance_array[2])) 

                        # if showing up and relevant - TP
                        if int(query_relevance_array[2]) > 0 and query_relevance_array[1] in docs_ids_new:
                            tp += 1   
                            try:
                                temp_ap = tp / (fp + tp)
                            except ZeroDivisionError:
                                temp_ap = 0
                            docs_ap.append(temp_ap)      

                            # treatment for ndcg
                            relevance_ndcg.append(float(query_relevance_array[2]))           
                    
                    elif int(query_relevance_array[0]) > query_n:
                        break
            
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

                # average precision
                try:
                    ap = sum(docs_ap)/len(docs_ap)
                except ZeroDivisionError:
                    ap = 0

                # ndcg
                ndcg_real = [relevance_ndcg[0]] + [relevance_ndcg[i]/(math.log2(i+1)) for i in range(1,len(relevance_ndcg))]
                ndcg_real = cumsum(ndcg_real)

                relevance_ndcg = sorted(relevance_ndcg)
                ndcg_ideal = [relevance_ndcg[0]] + [relevance_ndcg[i]/(math.log2(i+1)) for i in range(1,len(relevance_ndcg))]
                ndcg_ideal = cumsum(ndcg_ideal)
                
                ndcg = sum([r / i if i!=0 else 0 for r,i in zip(ndcg_real, ndcg_ideal)])

                #do the same but for calculating recall
                if i==0:
                    recall_10 = recall
                    precision_10 = precision
                    f_10 = f_score
                    ap_10 = ap
                    ndcg_10 = ndcg
                elif i==1:
                    recall_20 = recall
                    precision_20 = precision
                    f_20 = f_score
                    ap_20 = ap
                    ndcg_20 = ndcg
                elif i==2:
                    recall_50 = recall
                    precision_50 = precision
                    f_50 = f_score
                    ap_50 = ap
                    ndcg_50 = ndcg

                    # we also add the values to the array of 50 docs
                    self.mean_precision_array.append(precision)
                    self.mean_recall_array.append(recall)
                    self.mean_f_measure_array.append(f_score)
                    self.mean_ap_array.append(ap)
                    self.mean_ndcg_array.append(ndcg)
                    self.mean_latency_array.append(latency)
            
        print("Query: %d  %.3f %.3f %.3f \t %.3f %.3f %.3f \t   %.3f %.3f %.3f \t   %.3f %.3f %.3f \t   %.1f %.1f %.1f \t  %.0fms" % \
            (query_n, precision_10,precision_20,precision_50, recall_10, recall_20, recall_50, f_10, f_20, f_50 \
                ,ap_10,ap_20,ap_50, ndcg_10, ndcg_20, ndcg_50, latency*1000))

        return None

def usage():
    print("Usage: python3 main.py <tokenizer_mode: complex/simple> <chunksize:int> <ranking_mode:tf_idf/bm25> <analyze_flag:boolean>")