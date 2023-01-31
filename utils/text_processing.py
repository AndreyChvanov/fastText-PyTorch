import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

stopwords_list = stopwords.words('english')
lemma = nltk.wordnet.WordNetLemmatizer()


def preprocessing_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z]", " ", text)
    tokenize_text = []
    for word in word_tokenize(text):
        if word not in stopwords_list:
            word_lemma = lemma.lemmatize(word)
            tokenize_text.append(word_lemma)

    tokenize_text = " ".join(tokenize_text)
    return tokenize_text



