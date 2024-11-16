# vector_store.py
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add_vectors(self, vectors, texts):
        """
        Adds vectors to the Faiss index
        """
        self.index.add(np.array(vectors).astype('float32'))
        self.texts.extend(texts)

    def query(self, vector, k=1):
        """
        Queries the Faiss index and returns the top-k results
        """
        D, I = self.index.search(np.array([vector]).astype('float32'), k)
        return [(self.texts[i], D[0][idx]) for idx, i in enumerate(I[0])]

