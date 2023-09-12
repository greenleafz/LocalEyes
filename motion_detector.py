import cv2

# Configurable Variables
CONTOUR_AREA_THRESHOLD = 500  # Minimum contour area to detect motion


class MotionDetector:
    def __init__(self):
        """
        Initialize the motion detector class.
        """
        self.previous_frame = None

    def detect_motion(self, image):
        """
        Detect motion in the given image.

        Args:
        - image: Input image.

        Returns:
        - bool: True if motion detected, else False.
        """
        # Convert the current frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Applying Gaussian blur can help reduce noise and improve motion detection accuracy
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # If this is the first frame, store it and exit early
        if self.previous_frame is None:
            self.previous_frame = gray
            return False

        # Compute the absolute difference between the current and previous frame
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        
        # Threshold the difference image to binarize
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        # Dilation helps to fill in holes due to thresholding and consolidate detected changes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of the thresholded image (i.e., outlines of motion areas)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Update the previous frame for the next iteration
        self.previous_frame = gray

        # Check if any of the contours exceed our size threshold for detected motion
        for contour in contours:
            if cv2.contourArea(contour) > CONTOUR_AREA_THRESHOLD:
                return True

        return False
