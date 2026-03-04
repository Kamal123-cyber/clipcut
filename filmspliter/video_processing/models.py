import uuid
from django.db import models
from django.conf import settings


class EventVideo(models.Model):

    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="videos"
    )

    video_file = models.FileField(upload_to="videos/")
    title = models.CharField(max_length=255, blank=True)

    duration = models.FloatField(null=True, blank=True)

    is_indexed = models.BooleanField(default=False)

    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.uuid)


class FaceTrack(models.Model):

    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    video = models.ForeignKey(
        EventVideo,
        on_delete=models.CASCADE,
        related_name="face_tracks"
    )

    track_id = models.IntegerField()

    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.video_id} - {self.track_id}"


class FaceEmbedding(models.Model):

    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    track = models.ForeignKey(
        FaceTrack,
        on_delete=models.CASCADE,
        related_name="embeddings"
    )

    faiss_id = models.IntegerField()

    timestamp = models.FloatField()

    created = models.DateTimeField(auto_now_add=True)


class HighlightJob(models.Model):

    STATUS_CHOICES = (
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    )

    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )

    video = models.ForeignKey(
        EventVideo,
        on_delete=models.CASCADE
    )

    selfie = models.ImageField(upload_to="selfies/")

    result_video = models.FileField(
        upload_to="results/",
        null=True,
        blank=True
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="pending"
    )

    progress = models.IntegerField(default=0)

    created = models.DateTimeField(auto_now_add=True)