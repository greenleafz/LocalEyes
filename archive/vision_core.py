class Vision_Core:
    def __init__(self, motion_detector, video_buffer, preprocessor, model_inference):
        self.motion_detector = motion_detector
        self.video_buffer = video_buffer
        self.preprocessor = preprocessor
        self.model_inference = model_inference

    def capture_video(self, camera_handler):
        while camera_handler.is_opened():
            frame = camera_handler.capture_frame()
            if frame is not None:
                self.process_frame(frame)

    def process_frame(self, frame):
        if self.motion_detector.detect_motion(frame):
            objs = self.model_inference.detect_objects(frame)
            # Further logic to check if objs contains a person, save clip, etc.

    def process_clip(self, clip):
        pass  # your logic here

    def push_event(self, log, video):
        pass  # your logic here
