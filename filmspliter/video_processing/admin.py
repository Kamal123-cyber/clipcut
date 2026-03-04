from django.contrib import admin
from .models import EventVideo, HighlightJob, FaceTrack, FaceEmbedding
# Register your models here.
admin.site.register(EventVideo)
admin.site.register(HighlightJob)
admin.site.register(FaceTrack)
admin.site.register(FaceEmbedding)
