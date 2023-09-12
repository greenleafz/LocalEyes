from flask import Flask, render_template, Response
import cv2
from motion_detector import MotionDetector
from model_inference import ModelInference


app = Flask(__name__)

camera = cv2.VideoCapture(0)
motion_detector = MotionDetector()
model_inference = ModelInference()

frame_counter = 0
SAVE_EVERY_N_FRAMES = 1  # Change this value as needed
MAX_BUFFER_SIZE = 20 * 30  # Assuming 30fps, adjust accordingly
NO_PERSON_THRESHOLD = 10
frame_buffer = []

no_person_consecutive_frames = 0
detecting_motion = True

last_detected_person_boxes = []
last_detected_person_labels = []

def gen_frames():
    global frame_counter, frame_buffer
    global last_detected_person_boxes, last_detected_person_labels
    global no_person_consecutive_frames, detecting_motion

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame from camera.")
            break

        frame_counter += 1

        if detecting_motion and not motion_detector.detect_motion(frame):
            # If not detecting motion, just continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        # If we are here, it means motion has been detected
        cv2.putText(frame, "Motion Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        objs = model_inference.detect_objects(frame)
                
        # Filter out detected objects, keeping only persons with confidence above 40%
        person_objs = [obj for obj in objs if model_inference.labels.get(obj.id) == 'person' and obj.score >= 0.4]
                
        if person_objs:
            no_person_consecutive_frames = 0
            detecting_motion = False

            # Update and draw boxes
            last_detected_person_boxes = [obj.bbox for obj in person_objs]
            last_detected_person_labels = ['{}% person'.format(int(100 * obj.score)) for obj in person_objs]
            frame = model_inference.annotate_image(frame, person_objs)
            frame_buffer.append(frame)

            # Save buffer if full
            if len(frame_buffer) == MAX_BUFFER_SIZE:
                save_buffer_to_video(frame_buffer)
                frame_buffer = []
        else:
            no_person_consecutive_frames += 1

            # Draw the last known boxes on the frame
            for bbox, label in zip(last_detected_person_boxes, last_detected_person_labels):
                x0, y0, x1, y1 = map(int, [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax])
                frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                frame = cv2.putText(frame, label, (x0, y0+30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            # If we haven't seen a person for 10 consecutive frames, revert to motion detection
            if no_person_consecutive_frames >= NO_PERSON_THRESHOLD:
                detecting_motion = True
                last_detected_person_boxes = []
                last_detected_person_labels = []

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

def save_buffer_to_video(buffer):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640,480))  # Adjust size accordingly

    for frame in buffer:
        out.write(frame)

    out.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, ssl_context=('cert.pem', 'key.pem'))
