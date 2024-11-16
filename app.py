import streamlit as st
import re
from query_engine import process_documents, query_documents
import os

# Function to clean up repetitive or unnecessary parts of the query
def clean_repetitive_query(query):
    """Removes repetitive or redundant words/phrases from the query."""
    # Remove excessive repetitions (e.g., "drive so that it can drive" repeated many times)
    query = re.sub(r"(drive so that it can drive\s*)+", "drive", query)
    
   
    query = query.strip()
    
    return query

# Streamlit app title
st.title("SmartQuery: Extract Insights from the documents")

# Upload PDF files
uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

# Handle document upload
if uploaded_files:
    pdf_paths = []
    for uploaded_file in uploaded_files:
        unique_filename = f"uploaded_{uploaded_file.name}"
        with open(unique_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_paths.append(unique_filename)

    # Process documents
    process_documents(pdf_paths)
    st.success("Documents processed successfully!")

# Query input and results display
query = st.text_input("Ask a question based on the documents:")
if query:
    # Clean the query to remove repetitions
    cleaned_query = clean_repetitive_query(query)
    
    # Get response from the query engine
    response = query_documents(cleaned_query)
    
    # Display the response
    st.write(f"Answer: {response}")
