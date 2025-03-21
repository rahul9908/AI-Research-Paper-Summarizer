import streamlit as st
import pdfplumber
import pandas as pd
import torch
import networkx as nx
import re
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"  
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load ROUGE evaluation metric
rouge = evaluate.load("rouge")

# Function to extract text from PDFs with better filtering
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = []
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                lines = page_text.split("\n")
                cleaned_lines = [line.strip() for line in lines if len(line.split()) > 4]  # Remove short, meaningless lines
                text.extend(cleaned_lines)
    
    return " ".join(text)  # Proper sentence spacing

# Function to clean text by removing unwanted references, numbers, and symbols
def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Remove citations, numbers, and special characters
    text = re.sub(r"\[.*?\]", "", text)  
    text = re.sub(r"\b\d+\b", "", text)  
    text = re.sub(r"[^a-zA-Z\s.,]", "", text)  

    # Fix sentence spacing and punctuation
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.\.+", ".", text)  # Replace multiple dots
    text = re.sub(r",,", ",", text)  

    return text.lower()

# Extractive summarization using TextRank
def textrank_summarization(text, num_sentences=5):
    sentences = re.split(r'(?<=[.!?]) +', text)  # Improved sentence splitting

    if len(sentences) <= num_sentences:
        return " ".join(sentences)  # Return full text if too short

    vectorizer = TfidfVectorizer(stop_words="english")  
    X = vectorizer.fit_transform(sentences)
    
    similarity_matrix = cosine_similarity(X, X)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # Rank sentences and extract top ones
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = " ".join([s[1] for s in ranked_sentences[:num_sentences]])

    return summary

# Abstractive summarization using T5
def abstractive_summary(text):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Adjusted parameters for better summary generation
    summary_ids = model.generate(
        inputs, 
        max_length=200,  
        min_length=80,   
        length_penalty=1.5, 
        num_beams=8,      
        repetition_penalty=2.0,  
        temperature=0.8,  
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# ROUGE evaluation (ROUGE-1 and ROUGE-2 only)
def compute_rouge(reference, generated):
    scores = rouge.compute(predictions=[generated], references=[reference])
    return {"ROUGE-1": scores["rouge1"], "ROUGE-2": scores["rouge2"]}

# Streamlit UI
st.set_page_config(page_title="AI Research Paper Summarizer", layout="wide")

st.title("📄 AI Research Paper Summarizer")
st.write("Upload a research paper PDF, and the app will extract the text, summarize it, and evaluate the summary using ROUGE scores.")

uploaded_file = st.file_uploader("📂 Upload a research paper (PDF)", type="pdf")

if uploaded_file is not None:
    with st.spinner("🔍 Extracting text..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = clean_text(extracted_text)

    st.subheader("📜 Improved Extracted Text")
    st.text_area("", cleaned_text[:4000])  # Show more text for better analysis

    if st.button("📝 Summarize"):
        with st.spinner("🔍 Generating summaries..."):
            extractive_summary = textrank_summarization(cleaned_text)
            abstractive_summary_text = abstractive_summary(extractive_summary)

        st.subheader("🔍 Extractive Summary (TextRank)")
        st.write(extractive_summary)

        st.subheader("🧠 Abstractive Summary (T5 Model)")
        st.write(abstractive_summary_text)

        with st.spinner("📊 Computing ROUGE scores..."):
            rouge_scores = compute_rouge(extractive_summary, abstractive_summary_text)

        st.subheader("📊 ROUGE Scores")
        st.json(rouge_scores)

        st.markdown("""
        ### 🧐 What do these scores mean?
        - **ROUGE-1**: Measures how many important words match between summaries.
        - **ROUGE-2**: Measures how many important phrases match (2-word sequences).
        """)

st.markdown("---")
st.write(" Built by RAHUL TAMMALLA @copyrights ")
