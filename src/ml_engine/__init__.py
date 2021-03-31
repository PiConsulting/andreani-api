from typing import List
from .video_stream import process_video


def run_inference(video_path: str, debug=False):
    found_codes: List[str] = []
    for text in process_video(video_path, debug=debug):
        if text in found_codes:
            print("Code already in list:", text)
        else:
            print("New code:", text )
            found_codes.append(text)
    return found_codes
