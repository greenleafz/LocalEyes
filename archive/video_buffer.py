import cv2

class VideoBuffer:
    def __init__(self, buffer_size, output_path='output.mp4', fps=20.0, resolution=(640, 480)):
        self.buffer_size = buffer_size
        self.frames = deque(maxlen=buffer_size)
        
        # Video writer initialization
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'XVID' for AVI, 'mp4v' for MP4
        self.out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        self.is_recording = False

    def add_frame(self, frame):
        if not self.is_recording:
            print("Warning: Not recording. Call start_recording() first.")
        else:
            self.frames.append(frame)
            self.out.write(frame)

    def start_recording(self):
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False

    def save_video(self):
        for frame in self.frames:
            self.out.write(frame)
        self.frames.clear()  # Clear the frames once saved to video

    def release(self):
        self.out.release()

    def get_video(self):
        return list(self.frames)
