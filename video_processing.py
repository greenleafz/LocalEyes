import cv2
import os
import json
import datetime
import threading
from motion_detector import MotionDetector
from model_inference import ModelInference
import smtplib
from email.mime.text import MIMEText
from config import *

MAX_VIDEO_DURATION = 20 * 10  # 20 seconds at 10fps
NO_PERSON_SAVE_THRESHOLD = 2 * 10  # 2 seconds at 10fps
GRACE_PERIOD = 3 * 10  # 3 seconds at 10fps
TARGET_OBJECTS = {'person', 'cat', 'dog', 'bird'}
METADATA_FILE = './static/data/metadata.json'

motion_detector = MotionDetector()
model_inference = ModelInference()

frame_counter = 0
frame_buffer = []

no_object_consecutive_frames = 0
detecting_motion = True

last_detected_boxes = []
last_detected_labels = []

grace_frames_left = 0
saving_in_progress = False

detected_objects_in_video = set()


def gen_frames(camera):
    global frame_counter, frame_buffer
    global last_detected_boxes, last_detected_labels
    global no_object_consecutive_frames, detecting_motion

    video_started = False
    detected_objects = set()
    grace_frames_left = 0  # Initialize grace period frame counter


    while True:
        if saving_in_progress:
            continue
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame from camera.")
            break

        frame_counter += 1

        if detecting_motion and not motion_detector.detect_motion(frame) and grace_frames_left <= 0:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        # If we are here, it means motion has been detected or we're in the grace period.
        cv2.putText(frame, "Motion Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        objs = model_inference.detect_objects(frame)
        filtered_objs = [obj for obj in objs if model_inference.labels.get(obj.id) in TARGET_OBJECTS and obj.score >= 0.6]
                
        if filtered_objs or video_started:
            if filtered_objs:
                detected_objects.update([model_inference.labels.get(obj.id) for obj in filtered_objs])
                detected_objects_in_video.update(detected_objects)
                # print("Detected objects in video so far: ", detected_objects_in_video)


                video_started = True
                grace_frames_left = GRACE_PERIOD
                
                last_detected_boxes = [obj.bbox for obj in filtered_objs]
                last_detected_labels = ['{}% {}'.format(int(100 * obj.score), model_inference.labels.get(obj.id)) for obj in filtered_objs]
                frame = model_inference.annotate_image(frame, filtered_objs)
            else:
                grace_frames_left -= 1  # Decrease grace period frame count

            frame_buffer.append(frame)

            if len(frame_buffer) >= MAX_BUFFER_SIZE or grace_frames_left <= 0:
                if frame_buffer:
                    threading.Thread(target=save_buffer_to_video, args=(frame_buffer, list(detected_objects_in_video), 30)).start()
                    send_text_via_email(detected_objects_in_video)
                    frame_buffer = []
                    video_started = False
                    detected_objects.clear()
                    grace_frames_left = 0
                    detected_objects_in_video.clear()

        else:
            no_object_consecutive_frames += 1

            # Draw the last known boxes on the frame
            for bbox, label in zip(last_detected_boxes, last_detected_labels):
                x0, y0, x1, y1 = map(int, [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax])
                frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                frame = cv2.putText(frame, label, (x0, y0+30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            if no_object_consecutive_frames >= NO_PERSON_THRESHOLD:
                detecting_motion = True
                last_detected_boxes = []
                last_detected_labels = []

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def save_buffer_to_video(buffer, detections, max_videos=30):
    global saving_in_progress
    saving_in_progress = True

    # Define the directory
    directory = './static/data'
    
    # Check if directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'avc1' codec for .mp4 format

    # Define the filename using current datetime
    filename = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.mp4'
    filepath = os.path.join(directory, filename)

    out = cv2.VideoWriter(filepath, fourcc, 10.0, (640, 480))  # Adjust size accordingly

    for frame in buffer:
        out.write(frame)

    out.release()

    # Extract thumbnail right after saving the video
    thumbnail_filename = filename.replace('.mp4', '.jpg')  # Assuming thumbnails will be saved in .jpg format
    thumbnail_filepath = os.path.join(directory, thumbnail_filename)
    extract_thumbnail(filepath, thumbnail_filepath)

    # Update metadata
    update_metadata(filename, thumbnail_filename,detections)

    # If there are more than max_videos in the directory, delete the oldest ones
    all_videos = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')])
    while len(all_videos) > max_videos:
        # Delete the video
        os.remove(all_videos[0])
        # Delete its metadata
        delete_metadata_entry(all_videos[0].split('/')[-1])
        # Delete its corresponding thumbnail
        thumbnail_to_delete = all_videos[0].replace('.mp4', '.jpg')
        if os.path.exists(thumbnail_to_delete):
            os.remove(thumbnail_to_delete)
        
        all_videos.pop(0)
    saving_in_progress = False

def extract_thumbnail(video_path, thumbnail_path):
    """Extracts a thumbnail for a video."""
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if success:
        cv2.imwrite(thumbnail_path, image)

def update_metadata(video_filename, thumbnail_filename, detections):
    # Load existing metadata or initialize an empty dictionary
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Update or add new metadata entry
    metadata[video_filename] = {
        "thumbnail": thumbnail_filename,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "detections": detections if detections else []
        # You can add any other details you wish here
    }

    # Save updated metadata back to file
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)


def delete_metadata_entry(video_filename):
    """Removes a metadata entry corresponding to a video."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        if video_filename in metadata:
            del metadata[video_filename]

        # Save updated metadata back to file
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f)

def validate_metadata():
    """Validate metadata against actual files and remove invalid entries."""
    if not os.path.exists(METADATA_FILE):
        return

    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    directory = './static/data'
    actual_files = os.listdir(directory)

    # Check each file in metadata against actual files and remove if not found
    keys_to_delete = [key for key in metadata if key not in actual_files]
    for key in keys_to_delete:
        del metadata[key]

    # Save updated metadata back to file
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

def send_text_via_email(detections):
    # Define the content of the message
    msg = MIMEText('Motion has been detected:' + str(detections))
    
    # Email configuration
    from_email = "example@gmail.com"  # Replace with your email
    to_email = "phone-number@txt.att.net"  # Replace with the target phone's email-to-text address
    msg['Subject'] = 'New Video Alert'
    msg['From'] = from_email
    msg['To'] = to_email

    # Gmail SMTP configuration
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    gmail_password = "your-app-password"  # Replace with your email password

    # Establish a connection and send the message
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Upgrade the connection to encrypted SSL/TLS
        server.login(from_email, gmail_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Successfully sent email")
    except Exception as e:
        print("Error: unable to send email. Reason:", e)