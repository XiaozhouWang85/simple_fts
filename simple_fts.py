# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 21:50:59 

Allows full text seach for small datasets that does not justify overhead of 
Lucene, Solr or Elasticsearch

@author: Xiaozhou Wang
"""

from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

class fts_inmem:
 
    def __init__(self,max_df=0.05):
        self.inverted_index = defaultdict(list)
        self.__unique_id = 0
        self.vectorizer = CountVectorizer(max_df=max_df,min_df=0)
        self.transformer = TfidfTransformer(smooth_idf=False)
 
    def lookup(self, document):
        tokens=[word for word in self.analyze(document) if word not  in self.vectorizer.stop_words_]
        token_ids=[self.vectorizer.vocabulary_.get(token) for token in tokens]
        all_docs=[self.inverted_index.get(token_id) for token_id in token_ids]
        uniq_docs=list(set([doc for l in all_docs if l is not None for doc in l]))
        if len(uniq_docs)>0:
            query_doc=self.vectorizer.transform([document])
            raw_res=self.tfidf[uniq_docs]
            res=list(zip(uniq_docs,list(cosine_similarity(query_doc,raw_res)[0])))
            res_srt=sorted(res, key=lambda tup: tup[1],reverse=True)
        elif len(uniq_docs)==0:
            res_srt=[]
        return res_srt
    
    def create_index(self, documents):
        self.bow = self.vectorizer.fit_transform(documents)
        self.tfidf=self.transformer.fit_transform(self.bow)
        self.analyze = self.vectorizer.build_analyzer()
        self.inverted_index = defaultdict(list)
        self._create_inverted(documents)
    
    def _create_inverted(self, documents):
        for document in documents:
            tokens=[word for word in self.analyze(document) if word not  in self.vectorizer.stop_words_]
            for token in tokens:
                token_id=self.vectorizer.vocabulary_.get(token)
                if self.__unique_id not in self.inverted_index[token_id]:
                    self.inverted_index[token_id].append(self.__unique_id)
            self.__unique_id += 1

if __name__ == "__main__":
    index = fts_inmem(max_df=0.10)
    data=[
        'fairview park,no. 36,fairview park 9th street river  north,yuen long,new territories',
        'fairview park,no. 36,fairview park river north 9th street,yuen long,new territories',
        'hing sing building,no. 8-8a,un chau street,sham shui po,kowloon',
        'house 36,fairview park,9th street,river north,yuen long,new territories',
        'no. 51,south wall road,kowloon city,kowloon',
        'no. 24,san hong street,north,new territories',
        'no. 214-216,tung choi street,yau tsim mong,kowloon',
        "no. 4,gilman's bazaar,central and western,hong kong",
        'house 80,fairview park section l, 5th street,yuen long,new territories',
        'hong ning building,no. 162-166,cheung sha wan road,sham shui po,kowloon',
        'block 6,kambridge garden,no. 1,razor hill road,sai kung,new territories',
        'block 28,chun wah villas phase iii,no. 12,ma tong road, shap pat heung,yuen long,new territories',
        'chun wah villas phase 3 block 28,chun wah villas, phase iii,no. 12,ma tong road,yuen long,new territories',
        'no. 152-154,ma tau wai road,kowloon city,kowloon',
        'no. 5,canal road east,wan chai,hong kong'
    ]
    index.create_index(data)
    returned=index.lookup('sing')
    print(data[2])
