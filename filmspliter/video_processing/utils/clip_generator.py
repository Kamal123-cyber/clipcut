import os
import uuid
import subprocess
import json

from django.conf import settings


def get_video_duration(video_path: str) -> float:
    """Get exact video duration using ffprobe."""
    command = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0))


def generate_clip(video_path: str, start: float, end: float) -> dict | None:
    """
    Generate a video clip and thumbnail from the given time range.

    Returns a dict with:
        - clip_url: streamable MP4 URL
        - thumbnail_url: preview image URL
        - duration: clip duration in seconds
        - start / end: actual timestamps used

    Returns None if ffmpeg fails.
    """

    clip_id = uuid.uuid4()
    clip_name = f"{clip_id}.mp4"
    thumb_name = f"{clip_id}.jpg"

    clip_folder = os.path.join(settings.MEDIA_ROOT, "clips")
    thumb_folder = os.path.join(settings.MEDIA_ROOT, "thumbnails")
    os.makedirs(clip_folder, exist_ok=True)
    os.makedirs(thumb_folder, exist_ok=True)

    clip_path = os.path.join(clip_folder, clip_name)
    thumb_path = os.path.join(thumb_folder, thumb_name)

    # Clamp to actual video length so we never request frames past the end
    video_duration = get_video_duration(video_path)
    if video_duration > 0:
        start = max(0.0, min(start, video_duration - 1))
        end = min(end, video_duration)

    if end <= start:
        return None

    clip_duration = end - start

    # Re-encode to H.264/AAC for guaranteed browser compatibility.
    # -movflags +faststart moves the moov atom to the front so the clip
    # can start playing before it's fully downloaded (critical for streaming).
    clip_command = [
        "ffmpeg",
        "-y",
        "-ss", str(start),          # seek BEFORE -i for speed (input seek)
        "-i", video_path,
        "-t", str(clip_duration),   # duration-based trim is more reliable than -to after seek
        "-c:v", "libx264",
        "-preset", "fast",          # fast encode, good quality
        "-crf", "23",               # quality: 18=high, 28=low, 23=default
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",  # web-streamable
        "-avoid_negative_ts", "make_zero",
        clip_path
    ]

    result = subprocess.run(clip_command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ClipGenerator] ffmpeg error:\n{result.stderr}")
        return None

    # Generate thumbnail from the middle frame of the clip
    thumb_command = [
        "ffmpeg",
        "-y",
        "-ss", str(clip_duration / 2),
        "-i", clip_path,
        "-vframes", "1",
        "-q:v", "2",               # JPEG quality (2=best, 31=worst)
        "-vf", "scale=320:-1",     # 320px wide thumbnail
        thumb_path
    ]

    subprocess.run(thumb_command, capture_output=True)

    return {
        "clip_url": f"/media/clips/{clip_name}",
        "thumbnail_url": f"/media/thumbnails/{thumb_name}",
        "duration": round(clip_duration, 2),
        "start": round(start, 2),
        "end": round(end, 2),
    }