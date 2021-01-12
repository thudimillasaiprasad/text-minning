#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk as nlp
import wordcloud as wc
import matplotlib.pyplot as plt
import string
import warnings
warnings.filterwarnings("ignore")


# In[2]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nlp.download("wordnet")
nlp.download('stopwords')
nlp.download('punkt')

warnings.filterwarnings("ignore")


# ### Initialize Stemming & Lemmatizing objects

# In[3]:


st = nlp.PorterStemmer()
lem = nlp.WordNetLemmatizer()


# In[4]:


print("GOOSE STEMMING :", st.stem("Goose"))
print("GEESE STEMMING :", st.stem("Geese"))
print("GOOSE LEMMATIZING :", lem.lemmatize("Goose"))
print("GEESE LEMMATIZING :", lem.lemmatize("Geese"))


# In[5]:


print("MEANING STEMMING :", st.stem("Meaning"))
print("MEANNESS STEMMING :", st.stem("Meanness"))
print("MEANING LEMMATIZING :", lem.lemmatize("Meaning"))
print("MEANNESS LEMMATIZING :", lem.lemmatize("Meanness"))


# ###Create Word cloud

# In[6]:


corpus = "Climate change refers to the change in the environmental conditions of the Earth This happens due to many internal and external factors The climatic change has become a global concern over the last few decadesBesides, these climatic changes affect life on the earth in various waysThese climatic changes are having various impacts on the ecosystem and ecologyDue to these changes, a number of species of plants and animals have gone extinct."
print(corpus)


# ###Preprocess the text

# In[7]:


##Step 1: Lower case conversion

corpus=corpus.lower()

print("LOWER CASE TEXT :", corpus)


# In[8]:


##Step 2: Remove punctuation

punct = string.punctuation + "““\n"
text_without_punct = [char for char in corpus if char not in punct]
text_without_punct = "".join(text_without_punct)
print("TEXT WITHOUT PUNCTUATION :", text_without_punct)


# In[9]:


###Step 3: Remove stop-words

word_tokens = word_tokenize(text_without_punct)
print("WORD TOKENS :", word_tokens)
print("************************************************************************************************************************")
stop_words = set(stopwords.words('english'))
print("STOP WORDS :", stop_words)
print("************************************************************************************************************************")
word_tokens_without_stopwords = [w for w in word_tokens if not w in stop_words] 
print("WORD TOKENS WITHOUT STOP-WORDS :", word_tokens_without_stopwords)
print("************************************************************************************************************************")
text_without_stopwords = " ".join(word_tokens_without_stopwords)
print("TEXT WITHOUT STOP-WORDS :", text_without_stopwords)


# In[10]:


###Plot Word cloud

plt.imshow(wc.WordCloud(background_color="azure", normalize_plurals="true").generate(text_without_stopwords))


# In[ ]:




