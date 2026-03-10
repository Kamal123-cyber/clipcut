import cv2
import insightface
import numpy as np
from collections import defaultdict

from deep_sort_realtime.deepsort_tracker import DeepSort

from video_processing.models import EventVideo, FaceTrack, FaceEmbedding
from video_processing.utils.faiss_manager import get_video_faiss_index


# Cosine similarity threshold for re-identifying the same person across tracks
SAME_PERSON_THRESHOLD = 0.45

# Max embeddings stored PER APPEARANCE (not total per identity).
# Person appears 3 times → up to 3 embeddings × 3 appearances = 9 FAISS entries.
# This guarantees ALL appearances are searchable, not just the first.
MAX_EMBEDDINGS_PER_APPEARANCE = 3

# Process every N frames
FRAME_SKIP = 5

# Minimum IOU to accept a face-track match
MIN_IOU = 0.3

# Gap in seconds before a re-entry counts as a new appearance
NEW_APPEARANCE_GAP = 5.0


def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)


class IdentityBank:
    """
    Persists identities across broken DeepSort track IDs.
    When someone walks out of frame and back in, DeepSort gives a new track_id.
    IdentityBank re-links it to the same identity via embedding similarity.
    """

    def __init__(self, threshold=SAME_PERSON_THRESHOLD):
        self.threshold = threshold
        self.identities: dict[int, list[np.ndarray]] = {}
        self._next_id = 0

    def _norm(self, emb):
        return emb / (np.linalg.norm(emb) + 1e-6)

    def match_or_create(self, embedding: np.ndarray) -> int:
        embedding = self._norm(embedding)
        best_id, best_score = None, -1.0

        for iid, stored in self.identities.items():
            score = float(np.mean([np.dot(embedding, e) for e in stored]))
            if score > best_score:
                best_score = score
                best_id = iid

        if best_id is not None and best_score >= self.threshold:
            if len(self.identities[best_id]) < 20:
                self.identities[best_id].append(embedding)
            return best_id

        new_id = self._next_id
        self._next_id += 1
        self.identities[new_id] = [embedding]
        return new_id


class AppearanceTracker:
    """
    Tracks separate appearances for each identity.

    A new appearance is declared when an identity hasn't been seen
    for > NEW_APPEARANCE_GAP seconds. Each appearance gets its own
    embedding quota so all appearances end up in FAISS.

    Without this, the old MAX_EMBEDDINGS_PER_IDENTITY limit would fill
    up on the first appearance and store nothing for later ones.
    """

    def __init__(self, gap=NEW_APPEARANCE_GAP):
        self.gap = gap
        self._last_seen: dict[int, float] = {}
        self._appearance_idx: dict[int, int] = defaultdict(int)
        self._counts: dict[tuple, int] = defaultdict(int)

    def get_key(self, identity_id: int, timestamp: float) -> tuple:
        last = self._last_seen.get(identity_id)
        if last is not None and (timestamp - last) > self.gap:
            self._appearance_idx[identity_id] += 1
        self._last_seen[identity_id] = timestamp
        return (identity_id, self._appearance_idx[identity_id])

    def should_store(self, key: tuple) -> bool:
        return self._counts[key] < MAX_EMBEDDINGS_PER_APPEARANCE

    def record(self, key: tuple):
        self._counts[key] += 1


class VideoIndexer:

    def __init__(self):
        self.face_model = insightface.app.FaceAnalysis()
        self.face_model.prepare(ctx_id=0)
        self.tracker = DeepSort(max_age=30)

    def index_video(self, video_instance: EventVideo):

        video_path = video_instance.video_file.path
        video_uuid = str(video_instance.uuid)

        faiss_index = get_video_faiss_index(video_uuid)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        identity_bank = IdentityBank()
        appearance_tracker = AppearanceTracker()
        identity_to_db_track: dict[int, FaceTrack] = {}

        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % FRAME_SKIP != 0:
                frame_number += 1
                continue

            faces = [f for f in self.face_model.get(frame) if f.embedding is not None]
            detections = [
                ([int(f.bbox[0]), int(f.bbox[1]),
                  int(f.bbox[2] - f.bbox[0]), int(f.bbox[3] - f.bbox[1])], 1.0, "face")
                for f in faces
            ]

            tracks = self.tracker.update_tracks(detections, frame=frame)
            timestamp = frame_number / fps

            for track in tracks:
                if not track.is_confirmed():
                    continue

                l, t, r, b = track.to_ltrb()

                best_face, best_iou = None, 0.0
                for face in faces:
                    iou = bbox_iou([l, t, r, b], list(face.bbox))
                    if iou > best_iou:
                        best_iou = iou
                        best_face = face

                if best_face is None or best_iou < MIN_IOU:
                    continue

                embedding = best_face.embedding.astype("float32")
                identity_id = identity_bank.match_or_create(embedding)

                # Key = (identity_id, appearance_number)
                # Each appearance is tracked separately
                appearance_key = appearance_tracker.get_key(identity_id, timestamp)

                if not appearance_tracker.should_store(appearance_key):
                    continue

                if identity_id not in identity_to_db_track:
                    db_track = FaceTrack.objects.create(
                        video=video_instance,
                        track_id=identity_id
                    )
                    identity_to_db_track[identity_id] = db_track

                norm_emb = embedding / (np.linalg.norm(embedding) + 1e-6)
                faiss_id = faiss_index.add_embedding(norm_emb)

                FaceEmbedding.objects.create(
                    track=identity_to_db_track[identity_id],
                    faiss_id=faiss_id,
                    timestamp=timestamp
                )

                appearance_tracker.record(appearance_key)

            frame_number += 1

        cap.release()
        video_instance.is_indexed = True
        video_instance.save()
        faiss_index.save()

        print(f"[Indexer] Done. {len(identity_to_db_track)} unique identities.")