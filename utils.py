import logging
import nltk
import streamlit as st

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    # Silenciar logs molestos
    logging.getLogger('ccxt').setLevel(logging.WARNING)

@st.cache_resource
def init_nltk():
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
