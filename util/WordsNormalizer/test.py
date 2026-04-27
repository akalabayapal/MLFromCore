# import csv
# import nltk
import countvectorizerc
# import re
# from nltk.corpus import stopwords
# o = csv.reader(open('emails.csv','r'))
# stop_words = set(stopwords.words('english'))
# def removestop(s:str):
#     rs = ''
#     # [^...] means "everything except re.sub(r'[^a-zA-Z0-9% ]', '', text)
#     s= re.sub(r'[^a-zA-Z0-9%\s]', '', s)
#     for word in s.split(' '):
#         if not word in stop_words and word.strip() != '':
#             rs += ' '+word
  
#     return rs
# preprocessed = []
# for i in o:
#     ns = removestop(i[0][8:])
#     preprocessed.append(ns)
# print('preprocessing done...')
# import time
# t1 = time.time()
# o = countvectorizerc.CountVectorizer()
# i = o.fit(preprocessed)
# t2 = time.time()
# print('delta time:',t2-t1)

import pickle
# pickle.dump(o,open('wordsvector.pkl','wb'))


p:countvectorizerc.CountVectorizer = pickle.load(open('wordsvector.pkl','rb'))

h = p.transform(["hello 100 % free free free",'learn coding completely free'])
print(h)
for i in h[0][0]:
    print(p.getVocab()[i])







