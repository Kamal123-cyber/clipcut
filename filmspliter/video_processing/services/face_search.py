import insightface
import numpy as np
import cv2

from video_processing.models import FaceEmbedding
from video_processing.utils.faiss_manager import get_video_faiss_index


# Minimum cosine similarity to accept a match
SEARCH_THRESHOLD = 0.40

# How many FAISS neighbours to pull before filtering
FAISS_TOP_K = 50

# Minimum gap in seconds between returned clips.
# Set this to your clip duration (PAD_BEFORE + PAD_AFTER = 7s) so clips don't overlap.
# Do NOT set lower than your clip window or you'll return overlapping clips.
MIN_CLIP_GAP = 7.0


class FaceSearchService:

    def __init__(self):
        self.face_model = insightface.app.FaceAnalysis()
        self.face_model.prepare(ctx_id=0)

    def _get_query_embedding(self, image_path: str) -> np.ndarray | None:
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        faces = self.face_model.get(frame)
        if not faces:
            return None
        # Use the largest face in the selfie
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = largest.embedding.astype("float32")
        return emb / (np.linalg.norm(emb) + 1e-6)

    def search(self, image_path: str, video_uuid: str) -> list[FaceEmbedding]:
        """
        Returns one FaceEmbedding per distinct appearance of the person,
        deduplicated so clips don't overlap each other.
        """
        query_emb = self._get_query_embedding(image_path)
        if query_emb is None:
            print("[Search] No face detected in selfie.")
            return []

        faiss_index = get_video_faiss_index(video_uuid)

        if faiss_index.index.ntotal == 0:
            print(f"[Search] Empty FAISS index for video {video_uuid}.")
            return []

        # Pull more candidates than needed — we'll filter by threshold
        k = min(FAISS_TOP_K, faiss_index.index.ntotal)
        indices, scores = faiss_index.search(query_emb, k=k)

        print(f"[Search] Top scores: {scores[:5]}")  # helpful for threshold tuning

        matched_ids = [
            idx for idx, score in zip(indices, scores)
            if score >= SEARCH_THRESHOLD and idx >= 0
        ]

        if not matched_ids:
            print(f"[Search] No matches above threshold {SEARCH_THRESHOLD}.")
            return []

        # Fetch all matched embeddings for this video, sorted by time
        matched_embeddings = (
            FaceEmbedding.objects
            .filter(faiss_id__in=matched_ids, track__video__uuid=video_uuid)
            .select_related("track")
            .order_by("timestamp")
        )

        # Deduplicate: keep one result per appearance window.
        # MIN_CLIP_GAP should match your clip window length so clips don't overlap.
        # This correctly returns ALL appearances — first, second, third, etc.
        results = []
        last_timestamp = -MIN_CLIP_GAP

        for emb in matched_embeddings:
            if emb.timestamp - last_timestamp >= MIN_CLIP_GAP:
                results.append(emb)
                last_timestamp = emb.timestamp

        print(f"[Search] Returning {len(results)} appearances for video {video_uuid}.")
        return results