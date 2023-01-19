import streamlit as st
import pickle
import string
import nltk
import time
import base64
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('all-corpora')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

##


@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("nasa-Q1p7bh3SHj8-unsplash.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"]{{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
color: rgba(255, 255, 255);
}}
[id="it-s-summer"] {{
color: rgba(255, 255, 255);
}}
stMarkdownContainer.e16nr0p34 {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100vh;
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    background: #000;
}}

div.stButton > button{{
    width: 220px;
    height: 50px;
    border: none;
    outline: none;
    color: rgbd(255, 255, 255);
    background: rgbd(255, 255, 255);
    background-color: rgbd(255, 255, 255);
    cursor: pointer;
    position: relative;
    z-index: 0;
    border-radius: 10px;
}}

    button:before {{
    content: '';
    background: linear-gradient(45deg, #ff0000, #ff7300, #fffb00, #48ff00, #00ffd5, #002bff, #7a00ff, #ff00c8, #ff0000);
    position: absolute;
    top: -2px;
    left:-2px;
    background-size: 400%;
    z-index: -1;
    filter: blur(5px);
    width: calc(100% + 4px);
    height: calc(100% + 4px);
    animation: glowing 20s linear infinite;
    opacity: 0;
    transition: opacity .3s ease-in-out;
    border-radius: 10px;
    }}

    button:active {{
    color: rgbd(255, 255, 255); /* preserves the text color */
    border-color: transparent;
    background-color: white;
    }}

   button:active:after {{
        background: transparent;
    }}

    button:hover:before {{
    opacity: 1;
    }}

    button:after {{
    z-index: -1;
    content: '';
    position: absolute;
    width: 100%;
    height: 100%;
    background: #111;
    left: 0;
    top: 0;
    border-radius: 10px;
    }}

    @keyframes glowing {{
    0% {{ background-position: 0 0; }}
    50% {{ background-position: 400% 0; }}
    100% {{ background-position: 0 0; }}
    }}
    button:onclick{{
        disable hover
    }}
    stButton{{
         color: white;
    }}
    stButton:active{{
        color:rgbd(255, 255, 255);
    }}
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

##

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

    with st.spinner('Checking The Text...'):
        time.sleep(5)
    # Final Output
    if (result1 == 1) and (result2 == 1) and (result3 == 1) and (result4 == 1):
        st.header('The Submitted Text is :red[SPAM!]')
    else:
        st.header("The Submitted Text is :green[Not A SPAM!]")
