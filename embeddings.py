# embeddings.py
from sentence_transformers import SentenceTransformer

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text):
    """
    Converts text into embeddings using Sentence-BERT
    """
    embeddings = model.encode(text)
    return embeddings


