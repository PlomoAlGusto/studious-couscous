import logging
import nltk
import streamlit as st

def setup_logging():
    """Configura el logging para que no llene la consola de basura"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    # Silenciar logs ruidosos de librerías externas
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

@st.cache_resource
def init_nltk():
    """Inicializa NLTK una sola vez usando caché"""
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)