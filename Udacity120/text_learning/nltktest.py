# -*- coding: utf-8 -*-

import nltk

#nltk.download()

from nltk.corpus import stopwords
sw = stopwords.words("english")

print(len(sw))

from nltk.stem.snowball import SnowballStemmer
st = SnowballStemmer("english")

print(st.stem("fucking"))