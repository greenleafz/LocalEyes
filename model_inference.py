import os
import cv2

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference


# Configurable Variables
MODEL_PATH = 'model/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite'
LABEL_PATH = 'model/coco_labels.txt'
THRESHOLD = 0.1


class ModelInference:
    def __init__(self, model_path=MODEL_PATH, label_path=LABEL_PATH, threshold=THRESHOLD):
        """
        Initialize the model inference class.

        Args:
        - model_path: Path to the model file.
        - label_path: Path to the label file.
        - threshold: Detection threshold.
        """

        # Check if model and label files exist
        if not os.path.exists(model_path) or not os.path.exists(label_path):
            raise Exception("Model or label file not found.")
        
        # Initialize the TFLite interpreter and read labels
        try:
            self.interpreter = make_interpreter(model_path)
            self.interpreter.allocate_tensors()
            self.labels = read_label_file(label_path)
        except Exception as e:
            print(f"Error initializing the interpreter: {e}")
            raise
        
        self.inference_size = input_size(self.interpreter)
        self.threshold = threshold

    def detect_objects(self, image):
        """
        Detect objects in the given image.

        Args:
        - image: Input image.

        Returns:
        - objs: Detected objects.
        """
        # Convert and resize the image for inference
        cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, self.inference_size)

        # Run inference
        run_inference(self.interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(self.interpreter, self.threshold)

        return objs

    def annotate_image(self, image, objs):
        """
        Annotate the image with the detected objects.

        Args:
        - image: Input image.
        - objs: Detected objects.

        Returns:
        - cv2_im: Annotated image.
        """
        return append_objs_to_img(image, self.inference_size, objs, self.labels)


def append_objs_to_img(cv2_im, inference_size, objs, labels):
    """
    Append the detected objects to the image.

    Args:
    - cv2_im: Input image.
    - inference_size: Inference image size.
    - objs: Detected objects.
    - labels: Object labels.

    Returns:
    - cv2_im: Image with appended objects.
    """
    height, width, _ = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    return cv2_im
