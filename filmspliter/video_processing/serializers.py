from rest_framework import serializers
from .models import EventVideo


class EventVideoUploadSerializer(serializers.ModelSerializer):

    class Meta:
        model = EventVideo
        fields = [
            "uuid",
            "title",
            "video_file",
            "duration",
            "is_indexed",
            "created"
        ]

        read_only_fields = [
            "uuid",
            "duration",
            "is_indexed",
            "created"
        ]



class FaceSearchSerializer(serializers.Serializer):

    video_uuid = serializers.UUIDField()
    image = serializers.ImageField()