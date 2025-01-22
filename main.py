import os
import re
import requests
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

api_key = os.getenv('API_KEY')

def get_model():
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    chat_session = model.start_chat(history=[])
    return chat_session

def build_retrieval_system(documents):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)
    return vectorizer

def retrieve_relevant_passage(query, vectorizer, documents):
    query_vec = vectorizer.transform([query])
    doc_vectors = vectorizer.transform(documents)
    similarity = cosine_similarity(query_vec, doc_vectors).flatten()
    most_similar_idx = np.argmax(similarity)
    relevant_doc = documents[most_similar_idx]

    # Split the document into sentences
    sentences = relevant_doc.split('. ')
    sentence_vectors = vectorizer.transform(sentences)
    sentence_similarity = cosine_similarity(query_vec, sentence_vectors).flatten()
    
    # Get top 3 most similar sentences
    top_n = 3
    most_similar_sentence_indices = sentence_similarity.argsort()[-top_n:][::-1]
    most_similar_sentences = [sentences[idx] for idx in most_similar_sentence_indices]
    
    return ' '.join(most_similar_sentences)

def is_youtube_url(url):
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/.+$')
    return youtube_regex.match(url)

def is_website_url(url):
    website_regex = re.compile(
        r'^(https?://)?(www\.)?([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}(/.*)?$')
    return website_regex.match(url)

def get_chat_response(chat_session, message):
    response = chat_session.send_message(message)
    return response

def get_url_content(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def get_youtube_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = ' '.join([line['text'] for line in transcript])
    return text

def main():
    st.title("URL ChatBot")
    st.markdown("Chat with URLs")

    url = st.text_input("Enter a URL:")
    if st.button("Enter") or st.session_state.get("url_entered"):
        st.session_state.url_entered = True

    if url:
        chat_session = get_model()
        documents = []

        if is_youtube_url(url):
            st.success("This is a YouTube video URL.")
            video_id = url.split("=")[-1]
            try:
                transcript = get_youtube_transcript(video_id)
                st.text_area(f"Transcript Preview: ", transcript, height=200)
                documents.append(transcript)
            except Exception as e:
                st.error(f"Error fetching the transcript: {e}")
        elif is_website_url(url):
            st.success("This is a website URL.")
            try:
                content = get_url_content(url)
                st.text(f"Website Content Preview: {content[:600]}...")
                if st.button("Copy Content"):
                    st.text_area("Content: ", content, height=200)
                documents.append(content)
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching the website: {e}")
        else:
            st.error("This is not a valid URL.")

        if documents:
            vectorizer = build_retrieval_system(documents)
            if "messages" not in st.session_state:
                st.session_state.messages = []

            if prompt := st.chat_input("Ask something about the content:"):
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
                try:
                    relevant_passage = retrieve_relevant_passage(prompt, vectorizer, documents)
                    response = chat_session.send_message(f"{relevant_passage}\n\n{prompt}").text
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error("Request timed out, please check your connection!")
                    response = "error: " + str(e)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

if __name__ == "__main__":
    main()
