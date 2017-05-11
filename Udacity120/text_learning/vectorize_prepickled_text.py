# -*- coding: utf-8 -*-
import pickle

### read in data
word_data  = pickle.load( open("your_word_data.pkl", "rb") )
from_data  = pickle.load( open("your_email_authors.pkl", "rb") )

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words="english")
worddata_train_counts = count_vect.fit_transform(word_data)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(worddata_train_counts)
worddata_train_tf = tf_transformer.transform(worddata_train_counts)


print(worddata_train_tf.shape)
xxx = (count_vect.get_feature_names())
print(xxx[34596])
print(xxx[34597])