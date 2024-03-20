import cv2
import os
from vidaud1 import download_video

def extract_frames(video_path, output_directory='frames', frame_interval=6):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    interval_frames = frame_interval * cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract one frame every frame_interval seconds
        if frame_count % interval_frames == 0:
            frame_filename = os.path.join(output_directory, f'frame_{int(frame_count // interval_frames)}.jpg')
            cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    cap.release()

    print(f"Total frames extracted: {frame_count // interval_frames}")

# Example usage
video_url = 'https://www.youtube.com/watch?v=rs9AFEebHsk'
output_directory = r'C:\Users\HARSHITA KAMANI\Desktop\yt'

downloaded_video_path = download_video(video_url, output_directory)

if downloaded_video_path:
    extract_frames(downloaded_video_path, frame_interval=6)