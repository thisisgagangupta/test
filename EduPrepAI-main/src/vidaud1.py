from pytube import YouTube
from moviepy.editor import VideoFileClip
import os

def download_video(video_url, output_path='.', output_filename='downloaded_video.mp4'):
    try:
        # Create a YouTube object
        yt = YouTube(video_url)

        # Get the highest resolution stream available
        video_stream = yt.streams.get_highest_resolution()

        # Download the video to the specified output path and filename
        video_stream.download(output_path, output_filename)

        downloaded_path = os.path.join(output_path, output_filename)
        print(f"Video downloaded successfully to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def extract_audio(video_path, output_path='extracted_audio.wav'):
    try:
        # Create a VideoFileClip object
        video_clip = VideoFileClip(video_path)

        # Extract the audio
        audio = video_clip.audio
        audio.write_audiofile(output_path, codec='pcm_s16le')  # Specify codec to avoid certain issues

        print(f"Audio extracted successfully to {output_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
video_url = 'https://www.youtube.com/watch?v=rs9AFEebHsk'
output_directory = r'C:\Users\HARSHITA KAMANI\Desktop\yt'

downloaded_video_path = download_video(video_url, output_directory, output_filename='downloaded_video.mp4')

if downloaded_video_path:
    extract_audio(downloaded_video_path)