import faiss
import numpy as np
import os


class FaceFaissIndex:
    """
    Per-video FAISS index using Inner Product (cosine similarity on normalized vectors).
    Using IndexFlatIP instead of IndexFlatL2 — for normalized embeddings,
    inner product == cosine similarity, which is far better for face re-id.
    """

    def __init__(self, dimension=512, index_path="faiss_index.bin"):
        self.dimension = dimension
        self.index_path = index_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            # Inner Product index (cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)

    def _normalize(self, embedding):
        embedding = np.array(embedding).astype("float32")
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def add_embedding(self, embedding):
        embedding = self._normalize(embedding)
        vector = np.array([embedding]).astype("float32")
        self.index.add(vector)
        return self.index.ntotal - 1

    def search(self, embedding, k=10):
        embedding = self._normalize(embedding)
        vector = np.array([embedding]).astype("float32")

        # Cap k to how many embeddings exist
        k = min(k, self.index.ntotal)
        if k == 0:
            return [], []

        distances, indices = self.index.search(vector, k)
        return indices[0].tolist(), distances[0].tolist()

    def save(self):
        faiss.write_index(self.index, self.index_path)

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)


def get_video_faiss_index(video_uuid: str, dimension=512) -> FaceFaissIndex:
    """
    Returns a per-video FAISS index so searches never bleed across videos.
    Each video gets its own .bin file.
    """
    from django.conf import settings
    index_dir = os.path.join(settings.MEDIA_ROOT, "faiss_indexes")
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, f"{video_uuid}.bin")
    return FaceFaissIndex(dimension=dimension, index_path=index_path)