import cv2
import insightface
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort

from video_processing.models import EventVideo, FaceTrack, FaceEmbedding
from video_processing.utils.faiss_manager import FaceFaissIndex


class VideoIndexer:

    def __init__(self):

        # load face model
        self.face_model = insightface.app.FaceAnalysis()
        self.face_model.prepare(ctx_id=0)

        # tracker
        self.tracker = DeepSort(max_age=30)

        # FAISS
        self.faiss_index = FaceFaissIndex()

    def index_video(self, video_instance: EventVideo):

        video_path = video_instance.video_file.path

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_number = 0

        track_map = {}

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            faces = self.face_model.get(frame)

            detections = []

            for face in faces:

                x1, y1, x2, y2 = face.bbox.astype(int)

                w = x2 - x1
                h = y2 - y1

                detections.append(([x1, y1, w, h], 1.0, "face"))

            tracks = self.tracker.update_tracks(detections, frame=frame)

            for track in tracks:

                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                if track_id not in track_map:

                    db_track = FaceTrack.objects.create(
                        video=video_instance,
                        track_id=track_id
                    )

                    track_map[track_id] = db_track

                db_track = track_map[track_id]

                timestamp = frame_number / fps

                face_embedding = faces[0].embedding

                faiss_id = self.faiss_index.add_embedding(face_embedding)

                FaceEmbedding.objects.create(
                    track=db_track,
                    faiss_id=faiss_id,
                    timestamp=timestamp
                )

            frame_number += 1

        cap.release()

        video_instance.is_indexed = True
        video_instance.save()

        self.faiss_index.save()