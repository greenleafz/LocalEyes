# Importing required libraries and modules
from flask import Flask, render_template, Response, jsonify
import cv2
from video_processing import gen_frames, extract_thumbnail, delete_metadata_entry
import os
import json
from datetime import datetime, timedelta

# Configurable Variables
METADATA_FILE = 'static/data/metadata.json'
VIDEO_DIR = './static/data'

# Initializing Flask app and camera capture
app = Flask(__name__)
camera = cv2.VideoCapture(0)

@app.route('/video_feed')
def video_feed():
    """ Endpoint for the live video feed. """
    return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videos')
def videos():
    """ Endpoint to retrieve a list of videos. """
    videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')], reverse=True)[:3]
    return render_template('video_list.html', videos=videos)

@app.route('/video_list')
def video_list():
    """ Endpoint to get categorized video data based on timestamp. """
    video_metadata = get_video_data()

    categorized_data = {
        "Older": [],
        "Past 7 days": [],
        "Today": []
    }
    
    today = datetime.today()
    one_week_ago = today - timedelta(days=7)

    for video, data in video_metadata.items():
        video_timestamp = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
        details = {
            "filename": video,
            "timestamp": data["timestamp"],
            "detections": data["detections"]
        }

        if video_timestamp.date() == today.date():
            categorized_data["Today"].append(details)
        elif video_timestamp >= one_week_ago:
            categorized_data["Past 7 days"].append(details)
        else:
            categorized_data["Older"].append(details)
            
    return jsonify(categorized_data)

@app.route('/')
def index():
    """ Endpoint for the main page. """
    video_data = get_video_data()
    videos = sorted(list(video_data.keys()), reverse=True)[:3]
    return render_template('index.html', videos=videos, video_data=video_data)

@app.route('/history')
def history():
    """ Endpoint for the history page. """
    videos = os.listdir(VIDEO_DIR)
    videos = [video.replace('.mp4', '') for video in videos if video.endswith('.mp4')]
    return render_template('history.html', all_videos=videos)

@app.route('/delete_video/<video_name>', methods=['POST'])
def delete_video(video_name):
    """ Endpoint to delete a video based on its name. """
    try:
        video_path = os.path.join(VIDEO_DIR, f'{video_name}.mp4')
        thumbnail_path = os.path.join(VIDEO_DIR, f'{video_name}.jpg')

        # Deleting video and thumbnail
        os.remove(video_path) if os.path.exists(video_path) else None
        os.remove(thumbnail_path) if os.path.exists(thumbnail_path) else None

        # Deleting metadata entry
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
            metadata.pop(video_name + '.mp4', None)
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f)

        return {"status": "success", "message": "Video deleted successfully."}, 200
    except Exception as e:
        return {"status": "error", "message": f"Error deleting video: {str(e)}"}, 500

def get_video_data():
    """ Utility function to fetch video metadata. """
    if not os.path.exists(METADATA_FILE):
        return {}
    
    with open(METADATA_FILE, 'r') as file:
        try:
            data = json.load(file)
            return data
        except json.decoder.JSONDecodeError:
            return {}

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False) # Uncomment the ssl_context if needed.
