import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stremer=PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)# this function is spliting the phrase into words EXP: hello there , how are you ? ----------->['hello', 'there', ',', 'how', 'are', 'you', '?'] 
def stem(word):
    return stremer.stem(word.lower())# this to recognize any similar words EXP: ["organize","organizes","organizing"] ------------>['organ', 'organ', 'organ']
def bag_of_words(tokenized_sentence,all_words):
    """
    sentence=["hello","how","are","you"]
    words=["hi","hello","I","you","bye","thank","cool"]
    bag = [   0,     1,   0,   1,     0,      0,     0]
    """
    tokenized_sentence=[stem(w)for w in tokenized_sentence]
    bag = np.zeros(len(all_words),dtype=np.float32)#get zeros for the length of the words
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0#if the w is in the tokenized sentence give it a 1
    return bag

sentence=["hello","how","are","you"]
words=["hi","hello","I","you","bye","thank","cool"]
bag = bag_of_words(sentence,words)


