import streamlit as st
import pickle
import string
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('all-corpora')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer (2).pkl', 'rb'))
mnb = pickle.load(open('model (2).pkl', 'rb'))
etc = pickle.load(open('etc.pkl', 'rb'))
svc = pickle.load(open('svc.pkl', 'rb'))
voting = pickle.load(open('voting.pkl', 'rb'))

st.title("Email && SMS Spam Classifier")

input_message = st.text_area("Enter Your Email or SMS")

if st.button('Submit'):
    # 1. PREPROCESS

    transformed_text = transform_text(input_message)

    # 2. Vectorization

    vectorized = tfidf.transform([transformed_text])
    dense_input = vectorized.toarray()
    # 3a. first prediction
    result1 = mnb.predict(vectorized)[0]
    result2 = svc.predict(dense_input)[0]
    result3 = etc.predict(vectorized)[0]
    result4 = voting.predict(dense_input)[0]


    # Final Output
    if (result1 == 1) and (result2 == 1) and (result3 == 1) and (result4 == 1):
        st.header('The Submitted Text is :red[SPAM!]')
    else:
        st.header("The Submitted Text is :green[Not A SPAM!]")
