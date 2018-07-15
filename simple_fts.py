# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 21:50:59 

Allows full text seach for small datasets that does not justify overhead of 
Lucene, Solr or Elasticsearch

@author: Xiaozhou Wang
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class fts_inmem:
     
    def __init__(self,*args,**kwargs):
        self.vectorizer = TfidfVectorizer(*args,**kwargs)
        
    def create_index(self, documents, column='data'):
        self.column=column
        if isinstance(documents, list) or isinstance(documents,pd.core.series.Series):
            self.documents=pd.DataFrame(documents,columns=[column])
        else:
            self.documents=documents
        self.tfidf= self.vectorizer.fit_transform(self.documents[self.column])
        
    def _lookup(self, query):
        doc_vec=self.vectorizer.transform(query)
        terms_tfidf=self.tfidf[:,np.sum(doc_vec.toarray(),axis=0)>0] #Keep only columns of sparse matrix in query
        rows, _ = terms_tfidf.nonzero() 
        uniq_docs=list(set(rows))
        filter_df=self.documents.loc[uniq_docs,:]
        if len(uniq_docs)>0:
            raw_res=self.tfidf[uniq_docs]
            filter_df['match']=cosine_similarity(doc_vec,raw_res)[0]
            filter_df.sort_values('match',inplace=True,ascending=False)
        return filter_df
    
    def query(self, query):
        res=self._lookup(query)
        return res

if __name__ == "__main__":
    index = fts_inmem(max_df=0.90)
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
    returned=index.query(['north','fairview','kowloon'])
    print(returned)
    df=pd.DataFrame(data,columns=['My Text'])
    df=pd.concat([df,pd.DataFrame(np.random.random((15, 3)),columns=['Col 1','Col 2','Col 3'])]
            ,axis=1)
    index.create_index(df,column='My Text')
    returned=index.query(['north','fairview','kowloon'])
    print(returned)
