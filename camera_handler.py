import cv2

class CameraHandler:
    def __init__(self):
        """
        Initialize the CameraHandler by attempting to open the default webcam.
        
        Raises:
            Exception: If the webcam cannot be initialized.
        """
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Error initializing webcam.")
    
    def capture_frame(self):
        """
        Captures a single frame from the webcam.
        
        Returns:
            ndarray or None: The captured frame if successful, or None if the frame couldn't be captured.
        """
        ret, image = self.camera.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            return None
        return image
    
    def __del__(self):
        """
        Destructor for the CameraHandler. Ensures the webcam is properly released.
        """
        self.cleanup()

    def cleanup(self):
        """
        Releases the webcam if it's currently opened.
        """
        if self.camera.isOpened():
            self.camera.release()
            print("Webcam released.")

    def is_opened(self):
        """
        Checks if the webcam is currently opened and operational.
        
        Returns:
            bool: True if the webcam is opened, otherwise False.
        """
        return self.camera.isOpened()
