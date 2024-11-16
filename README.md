# Content Engine: SmartQuery System  

## Overview  
The **Content Engine: SmartQuery System** is a robust application designed to extract insights from uploaded PDF documents using advanced natural language processing (NLP) techniques and a **local large language model (LLM)**. This system simplifies document processing and provides accurate answers to user queries by leveraging embeddings, vector storage, and an efficient local LLM for processing queries.  

## Motivation  
Extracting meaningful information from large volumes of documents is a common challenge in industries such as legal, financial, and research. The SmartQuery System addresses this by automating the extraction and query-answering process, reducing manual effort and increasing efficiency.  

## Methodology  
The SmartQuery System operates through the following steps:  

1. **Document Upload**  
   - Users upload PDF documents via a Streamlit interface.  
   - Uploaded files are pre-processed and converted into machine-readable text.  

2. **Text Embedding Generation**  
   - Extracted text is converted into embeddings using the **Sentence-BERT model (`all-MiniLM-L6-v2`)** for semantic understanding.  

3. **Vector Storage**  
   - The embeddings and corresponding document text are stored in a Faiss-based vector index for efficient similarity searches.  

4. **Query Processing**  
   - User queries are cleaned and converted into embeddings.  
   - The system retrieves the most relevant document sections using similarity metrics.  

5. **Answer Generation**  
   - Context from the relevant sections is passed to the **local LLM model (based on GPT-Neo)** to generate tailored responses efficiently, without relying on external cloud-based APIs.  

## Repository Content  

- **`app.py`**  
   The main Streamlit application for user interaction. Handles file uploads, document processing, and query input/output.  

- **`query_engine.py`**  
   Core logic for querying documents, including text extraction, embeddings generation, and interaction with the vector store.  

- **`llm.py`**  
   Handles the **local LLM model (GPT-Neo)** initialization and response generation.  

- **`vector_store.py`**  
   Implements a Faiss-based vector store for efficient document embedding storage and retrieval.  

- **`embeddings.py`**  
   Contains methods for generating embeddings using Sentence-BERT.  

- **`requirements.txt`**  
   Lists all Python dependencies required to run the application.  

## Dataset  
The system supports any PDF documents. Users can process their own documents for queries.  

## Installation  

1. **Clone the Repository**  

   ```bash  
   git clone https://github.com/ReetikaChavan/Content-Engine.git  
   cd Content-Engine  
