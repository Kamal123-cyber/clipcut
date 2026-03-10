import cv2
import os
import uuid

from django.conf import settings
from django.db import transaction

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

from .serializers import EventVideoUploadSerializer, FaceSearchSerializer
from .models import EventVideo
from .utils.video_indexer import VideoIndexer
from .services.face_search import FaceSearchService
from .utils.clip_generator import generate_clip

CLIP_PAD_BEFORE = 3
CLIP_PAD_AFTER = 4


class UploadVideoAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = EventVideoUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        with transaction.atomic():
            video_instance = serializer.save(user=request.user)
            print(video_instance, 'video uploaded')
            # Read duration while we still have the cap open
            cap = cv2.VideoCapture(video_instance.video_file.path)
            print(cap, 'video uploaded')
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(fps, 'video uploaded1')
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(frame_count, 'video uploaded2')
            cap.release()

            duration = (frame_count / fps) if fps > 0 else 0
            print(duration, 'video duration')
            video_instance.duration = duration
            video_instance.save()

            def start_indexing():
                try:
                    indexer = VideoIndexer()
                    indexer.index_video(video_instance)
                except Exception as e:
                    print(f"[Indexer] Failed for video {video_instance.uuid}: {e}")

            transaction.on_commit(start_indexing)

        return Response(
            {
                "message": "Video uploaded. Indexing started.",
                "video_uuid": str(video_instance.uuid),
                "duration": video_instance.duration,
            },
            status=status.HTTP_201_CREATED,
        )


class SearchFaceAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FaceSearchSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image = serializer.validated_data["image"]
        video_uuid = serializer.validated_data["video_uuid"]

        try:
            video = EventVideo.objects.get(uuid=video_uuid, user=request.user)
        except EventVideo.DoesNotExist:
            return Response({"error": "Video not found."}, status=status.HTTP_404_NOT_FOUND)

        if not video.is_indexed:
            return Response(
                {"error": "Video is still being indexed. Please try again shortly."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        file_name = f"{uuid.uuid4()}.jpg"
        save_path = os.path.join(settings.MEDIA_ROOT, "search", file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb+") as f:
            for chunk in image.chunks():
                f.write(chunk)

        service = FaceSearchService()
        matches = service.search(save_path, str(video_uuid))

        if not matches:
            return Response({"video_uuid": str(video_uuid), "total_clips": 0, "clips": []})

        video_path = video.video_file.path
        clips = []

        for match in matches:
            start = max(0.0, match.timestamp - CLIP_PAD_BEFORE)
            end = match.timestamp + CLIP_PAD_AFTER
            result = generate_clip(video_path, start, end)
            if result:
                clips.append({"timestamp": round(match.timestamp, 2), **result})

        clips.sort(key=lambda c: c["timestamp"])

        return Response({"video_uuid": str(video_uuid), "total_clips": len(clips), "clips": clips})