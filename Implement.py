import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
import tldextract as tde
from nltk import ngrams


st.set_page_config(page_title="DGA-Based Malicious Domain Detection", page_icon=":chart_with_upwards_trend:", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

#Load Model dan Load Word2vec
model_name = 'TRIGRAM_BIGRU_MULTICLASS_NEW'
w2v_name = 'trigram_word2vec_new'
best_model = tf.keras.models.load_model('./models/' + model_name+ '.h5', compile=False)
word2vec = Word2Vec.load('./models/'+w2v_name+'.model')

#this is the preprocess
def n_grams(word, n):

    # We can't find n-grams if the word has less than n letters.
    if n > len(word):
        return []

    output = []
    start_idx = 0
    end_idx = start_idx + n

    # Grab all n-grams except the last one
    while end_idx < len(word):
        n_gram = word[start_idx:end_idx]
        output.append(n_gram)
        start_idx = end_idx - 1
        end_idx = start_idx + n

    # Grab the last n-gram
    last_n_gram_start = len(word) - n
    last_n_gram_end = len(word)
    output.append(word[last_n_gram_start:last_n_gram_end])

    return output

extract = tde.TLDExtract(
include_psl_private_domains=True,
# suffix_list_urls=["file:///D:/THESIS_LAB/Dataset/public_suffix_list.txt"], 
cache_dir='./cache_tld/',
fallback_to_snapshot=False)

def extract_tld(domain):
    ext = extract(domain)
    return ext

def Preprocess(data):

    #Remove Subdomain
    clean_domains = []
    hostname = data['Host Name'].tolist()
    for dom in hostname:
        tmp = extract_tld(dom.lower())
        dms = tmp.domain+'.'+tmp.suffix
        clean_domains.append(dms)
    # return clean_domains

    # Creating the Trigram model
    totalRecord = len(clean_domains)
    corpus =[]
    for i, dom in enumerate(clean_domains):
            trigram = ["".join(k1) for k1 in list(ngrams(dom.lower(),n=3))] 
            corpus.append(trigram)

    X = np.zeros([len(corpus), 75], dtype=np.int32)
    for i, sentence in enumerate(corpus):
        #print(sentence)
        for t, word in enumerate(sentence):
            X[i, t] = word2vec.wv.key_to_index[word]
    
    return X
    

t1, t2 = st.columns((0.07,1)) 

t1.image('images/logo.png', width = 120)
t2.title("Domain Generated Algorithm (DGA) Based Malicious Domain Detection")


uploaded_file = st.file_uploader("Upload DNS Query Log (csv)", type=["csv"])
st.write('DNS Query Log didapat menggunakan tools [DNSQuery Sniffer](https://www.nirsoft.net/utils/dnsquerysniffer-x64.zip)')

if uploaded_file is not None:
    with st.spinner("Please wait...Detecting DGA Domain..."):
        dnscap = pd.read_csv(uploaded_file)
        #Open DNS Capture Data
        # Load data

        # data_home = '../DNSCapture/'
        # dnscap = pd.read_csv(data_home+'10.107.21.162_10012023.csv', encoding='ISO-8859-1', sep=',')
        # dnscap = pd.read_csv(data_home+'10.107.1.232_27122022.csv', encoding='ISO-8859-1', sep=',')
        print(len(dnscap))
        print(dnscap.columns)
        #Remove NXDomain
        # dnscap.drop(dnscap.index[dnscap["Response Code"] == 'Name Error'])

        X = Preprocess(dnscap)

        Y_pred = best_model.predict(X)
        Y_pred = np.argmax(Y_pred,axis=1)

        dga_labels_dict = {'normal':0, 'bamital': 1, 'banjori':2, 'bedep':3, 'chinad':4, 'conficker':5, 'corebot':6, 'cryptolocker':7, 'dnschanger':8, 'dyre':9, 'emotet':10, 'gozi':11,'locky':12, 'matsnu':13, 'monerominer':14, 'murofet':15, 'mydoom':16, 'padcrypt':17, 'pandabanker':18, 'qakbot':19,'rovnix':20, 'sisron':21, 'sphinx':22, 'suppobox':23,'sutra':24, 'symmi':25,'szribi':26, 'tinynuke':27, 'torpig':28, 'vidro':29,'virut':30}

        #reverse
        inv_label = {v: k for k, v in dga_labels_dict.items()}
        label_pred = list(map(inv_label.get, Y_pred))
        pred = {'DGA Domain?': label_pred}
        col_pred = pd.DataFrame(pred)

        new_df = pd.concat([dnscap,col_pred], axis=1)

        #swap columns
        cols = list(new_df.columns)
        a, b = cols.index('Query ID'), cols.index('DGA Domain?')
        cols[b], cols[a] = cols[a], cols[b]
        new_df = new_df[cols]
        
        st.dataframe(new_df)