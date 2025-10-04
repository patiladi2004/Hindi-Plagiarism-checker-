"""
Hindi Plagiarism Checker (single-file project)
Modified: Uses fixed thresholds for decisions
"""

# -------------------------
# Imports and utilities
# -------------------------
import os
import re
from typing import List

import numpy as np
import pandas as pd

import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# sentence-transformers for embeddings
from sentence_transformers import SentenceTransformer

# small Hindi stopword list (starter)
HINDI_STOPWORDS = set([
    'और','का','के','को','पर','ही','है','था','थे','हैं','यह','ये','उन','तथा','कि','से','में','भी','नहीं','या','तो','जो','में', 'एक'
])

# -------------------------
# Simple preprocessing for Hindi
# -------------------------
def preprocess_hindi(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.replace('\u200d', ' ')
    text = re.sub(r"[^\u0900-\u097F0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [tok for tok in text.split() if tok not in HINDI_STOPWORDS]
    return ' '.join(tokens)

# -------------------------
# Sample dataset writer (small)
# -------------------------
def write_sample_dataset(path='hindi_plagiarism_dataset.csv'):
    sample = [
        (1, 'भारत एक सुंदर देश है और यहाँ की विविधता अद्भुत है', 'भारत एक खूबसूरत देश है जहाँ विविधता बहुत है', 1),
        (2, 'मौसम आज बहुत सुहावना है', 'कल की परीक्षा महत्वपूर्ण होगी', 0),
        (3, 'विज्ञान ने मानव जीवन में क्रांति ला दी है', 'विज्ञान ने जिंदगी में बड़े परिवर्तन किए हैं', 1),
        (4, 'राम और श्याम बाजार गए', 'रवि और मोहन पार्क में खेले', 0),
        (5, 'कंप्यूटर विज्ञान में मशीन लर्निंग का उपयोग बढ़ रहा है', 'मशीन लर्निंग का उपयोग कंप्यूटर विज्ञान के क्षेत्र में बढ़ रहा है', 1)
    ]
    df = pd.DataFrame(sample, columns=['id','doc1','doc2','label'])
    df.to_csv(path, index=False)
    return path

# -------------------------
# Classical model: TF-IDF + Cosine similarity
# -------------------------
class TFIDFPlagiarism:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer='word', tokenizer=lambda x: x.split(), lowercase=False)

    def fit_transform_pair(self, doc1: str, doc2: str):
        X = self.vectorizer.fit_transform([doc1, doc2])
        return X

    def similarity(self, doc1: str, doc2: str) -> float:
        X = self.fit_transform_pair(doc1, doc2)
        sim = cosine_similarity(X[0], X[1])[0][0]
        return float(sim)

# -------------------------
# Embedding-based model
# -------------------------
class EmbeddingPlagiarism:
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def similarity(self, doc1: str, doc2: str) -> float:
        emb = self.model.encode([doc1, doc2], convert_to_numpy=True)
        a, b = emb[0], emb[1]
        dot = np.dot(a, b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(dot/denom)

# -------------------------
# Evaluation utilities
# -------------------------
from sklearn.metrics import precision_recall_fscore_support

def evaluate_models_on_df(df: pd.DataFrame):
    tf = TFIDFPlagiarism()
    emb = EmbeddingPlagiarism()

    tf_sims = []
    emb_sims = []
    labels = []

    for _, row in df.iterrows():
        d1 = preprocess_hindi(row['doc1'])
        d2 = preprocess_hindi(row['doc2'])
        labels.append(int(row['label']))
        try:
            tf_sims.append(tf.similarity(d1, d2))
        except Exception:
            tf_sims.append(0.0)
        try:
            emb_sims.append(emb.similarity(row['doc1'], row['doc2']))
        except Exception:
            emb_sims.append(0.0)

    # --- fixed thresholds ---
    TFIDF_THRESHOLD = 0.3
    EMBEDDING_THRESHOLD = 0.75

    tf_preds = [1 if s >= TFIDF_THRESHOLD else 0 for s in tf_sims]
    emb_preds = [1 if s >= EMBEDDING_THRESHOLD else 0 for s in emb_sims]

    tf_prec, tf_rec, tf_f1, _ = precision_recall_fscore_support(labels, tf_preds, average='binary', zero_division=0)
    emb_prec, emb_rec, emb_f1, _ = precision_recall_fscore_support(labels, emb_preds, average='binary', zero_division=0)

    return {
        'tf': {'precision':tf_prec, 'recall':tf_rec, 'f1':tf_f1, 'threshold':TFIDF_THRESHOLD},
        'emb': {'precision':emb_prec, 'recall':emb_rec, 'f1':emb_f1, 'threshold':EMBEDDING_THRESHOLD},
    }

# -------------------------
# Streamlit GUI
# -------------------------
st.set_page_config(page_title='Hindi Plagiarism Checker', layout='wide')

st.title('Hindi Plagiarism Checker — TFIDF (NLTK-style) vs Embedding (LLM-style)')

st.markdown('''
This demo compares a classical TF-IDF + cosine similarity approach with a dense embedding approach
(using a multilingual sentence-transformer) for Hindi text similarity / plagiarism detection.
''')

# Create sample dataset if not present
if not os.path.exists('sample_hindi_dataset.csv'):
    write_sample_dataset('sample_hindi_dataset.csv')

# Sidebar controls
st.sidebar.header('Options')
mode = st.sidebar.selectbox('Run mode', ['Single comparison', 'Batch evaluate (CSV sample)'])

if mode == 'Single comparison':
    st.subheader('Compare two documents')
    doc1 = st.text_area('Document 1', value='भारत एक सुंदर देश है और यहाँ की विविधता अद्भुत है')
    doc2 = st.text_area('Document 2', value='भारत एक खूबसूरत देश है जहाँ विविधता बहुत है')
    run_btn = st.button('Compare')

    if run_btn:
        d1_p = preprocess_hindi(doc1)
        d2_p = preprocess_hindi(doc2)

        tf_model = TFIDFPlagiarism()
        emb_model = EmbeddingPlagiarism()

        with st.spinner('Computing similarities...'):
            tf_score = tf_model.similarity(d1_p, d2_p)
            emb_score = emb_model.similarity(doc1, doc2)

        st.metric('TF-IDF cosine similarity', f'{tf_score:.4f}')
        st.metric('Embedding cosine similarity', f'{emb_score:.4f}')

        st.write('Preprocessed Document 1:', d1_p)
        st.write('Preprocessed Document 2:', d2_p)

        # --- fixed thresholds ---
        TFIDF_THRESHOLD = 0.3
        EMBEDDING_THRESHOLD = 0.75

        st.subheader("Decision")
        st.write('TF-IDF →', 'Plagiarized/similar' if tf_score >= TFIDF_THRESHOLD else 'Not plagiarized')
        st.write('Embedding →', 'Plagiarized/similar' if emb_score >= EMBEDDING_THRESHOLD else 'Not plagiarized')

elif mode == 'Batch evaluate (CSV sample)':
    st.subheader('Batch evaluate using sample_hindi_dataset.csv or upload your own CSV')
    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv('sample_hindi_dataset.csv')

    st.write('Dataset preview:')
    st.dataframe(df.head())

    if st.button('Run evaluation'):
        with st.spinner('Running evaluation (may download embedding model)...'):
            results = evaluate_models_on_df(df)
        st.success('Done')
        st.write('Results:')
        st.write('TF-IDF (classical):', results['tf'])
        st.write('Embedding (LLM-style):', results['emb'])

        st.markdown('**Notes**: thresholds are fixed (TF-IDF=0.3, Embedding=0.75). For a real system, tune thresholds on a dev set and use cross-validation.')

st.markdown('---')
st.write('Project created as a teaching starter kit — expand dataset and preprocessing for better results.')

if __name__ == '__main__':
    pass