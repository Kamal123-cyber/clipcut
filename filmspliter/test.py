from video_processing.models import EventVideo
from video_processing.utils.video_indexer import VideoIndexer

video = EventVideo.objects.first()
print(video)
indexer = VideoIndexer()

indexer.index_video(video)