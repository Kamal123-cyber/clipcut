import faiss
import numpy as np
import os


class FaceFaissIndex:

    def __init__(self, dimension=512, index_path="faiss_index.bin"):
        self.dimension = dimension
        self.index_path = index_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(dimension)

    def add_embedding(self, embedding):
        """
        Add a face embedding to the FAISS index
        """

        vector = np.array([embedding]).astype("float32")

        self.index.add(vector)

        return self.index.ntotal - 1  # return FAISS id

    def search(self, embedding, k=5):
        """
        Search for closest embeddings
        """

        vector = np.array([embedding]).astype("float32")

        distances, indices = self.index.search(vector, k)

        return indices[0], distances[0]

    def save(self):
        """
        Save index to disk
        """
        faiss.write_index(self.index, self.index_path)

    def load(self):
        """
        Reload index
        """
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)