import pickle
from nltk.corpus import wordnet as wn
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import conditionList
import drugname

tokenizer = RegexpTokenizer('\w+|^[\d\.]+|[^0-9]|\S+') # Tokenizer object
lemma = WordNetLemmatizer() # Lemmatization object
punctuations = '''!()-,[]{};:’“”'"\,,<>./?@#$%^&*_~''' # punctuation 
stopwords = stopwords.words('english') # obtain nltk ver. stopwords
drugsName = drugname.drugname() # list of drugs name
conditions = conditionList.condition() # list of medical condition
extra = ['would','will','also']

punctuations_list = [p for p in punctuations] # create punctuation list

# Remove stopwords that may remove negative sentiment 
remove_stopwords = ['not','no','very',"don't","ain",'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't","won't", 'wouldn', "wouldn't"]    
for r in remove_stopwords:
    stopwords.remove(r)

# transform effectiveness into numeric/ ordinal variable into 3 levels as scholar papers suggested
effectiveness_mapping = {'Highly Effective': 2, 'Considerably Effective':2, "Ineffective":0, 'Marginally Effective':1,'Moderately Effective':1}


# Check stopwords
unwanted_word = stopwords + punctuations_list +extra + drugsName + conditions # Remove influence of drug used and medical terms of condition

with open('vectorizer.pickle','rb') as v_file:
    vectorizer = pickle.load(v_file)

def hasNumbers(inputString):
    return not(bool(re.search(r'\d', inputString)))

def word_extraction(text):
    word_list = tokenizer.tokenize(text) # tranform word in list
    useWord_list =[]
    for word in word_list:
        word = word.lower() # set word to lowercase 
        if (word not in unwanted_word) & (len(word)>2) & (hasNumbers(word)): # remove unwanted words / words with single character / words with dose amount
            word = lemma.lemmatize(word,wn.NOUN) # lemmatize by noun
            word = lemma.lemmatize(word,wn.VERB) # lemmatize by verb
            word = lemma.lemmatize(word,wn.ADJ) # lemmatize by adj
            useWord_list.append(word)
    text = ' '.join(useWord for useWord in useWord_list) # recreate the text sentence for count vectorizing
    return text    

def text_preprocess(df, vectorizer,labelCol='effectiveness',textCol1='commentsReview',textCol2='benefitsReview'): # input data to pre-process with the trained vectorizer
    df['effectiveness_ordinal'] = df[labelCol].map(effectiveness_mapping)
    df.drop(columns=[labelCol],inplace=True)
    df.reset_index(inplace=True,drop=True)
    textSeries1 = df[textCol1]
    textSeries2 = df[textCol2] 
    label = df[['effectiveness_ordinal']] # obtain the label
    textSeries = textSeries1 + textSeries2 # combine the two columns with text comments
    textSeries = list(textSeries.map(word_extraction)) # words extraction and cleaning    
    count_vector = vectorizer.transform(textSeries).toarray() # obtain word count per combined comment
    feature = vectorizer.get_feature_names() # obtain words
    vector_df = pd.DataFrame(data=count_vector, columns=feature) # build dataframe for storing word vector
    df = pd.concat([vector_df,label],axis=1) # concat with original dataset
    return df
