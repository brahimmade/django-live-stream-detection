## Load model
```python
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', force_reload=True) 
imgs = ['0001.jpeg', '0002.jpeg']  # batch of images
results = model(imgs)
```

## cuda available
```python
# check if cuda available
>>> import torch
>>> torch.cuda.is_available()
True
```
## Detector 

```python
import torch
import numpy as np
import cv2
from time import time

class OD:

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
    
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()
    
        while True:
        
            ret, frame = cap.read()
            assert ret
            
            frame = cv2.resize(frame, (640,640))
            
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            #print(f"Frames Per Second : {fps}")
            
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv5 Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()
# Create a new object and execute.
detector = OD(capture_index=0, model_name='320_yolo_400.pt')
detector()
```

## Capture screen

```python
import time

import cv2
import mss
import numpy

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = sct.monitors[0]

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))

        # Display the picture
        cv2.imshow("OpenCV/Numpy normal", img)

        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

        print("fps: {}".format(1 / (time.time() - last_time)))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

```

results list received:
```python
results_list = [{'xmin': 1.9465026855, 'ymin': 62.251083374, 'xmax': 334.8536682129, 'ymax': 496.0778808594, 'confidence': 0.741962254, 'class': 59, 'name': 'bed'}, {'xmin': 1123.8736572266, 'ymin': 0.5675125122, 'xmax': 1279.8072509766, 'ymax': 167.1142883301, 'confidence': 0.6283448935, 'class': 62, 'name': 'tv'}, {'xmin': 1132.8088378906, 'ymin': 496.037109375, 'xmax': 1263.6857910156, 'ymax': 603.0759277344, 'confidence': 0.3710236847, 'class': 73, 'name': 'book'}, {'xmin': 540.3387451172, 'ymin': 0.0, 'xmax': 692.0639648438, 'ymax': 331.0688171387, 'confidence': 0.3388776481, 'class': 0, 'name': 'person'}, {'xmin': 0.7673721313, 'ymin': 376.2200927734, 'xmax': 216.818572998, 'ymax': 706.1331787109, 'confidence': 0.2838314772, 'class': 0, 'name': 'person'}, {'xmin': 98.2866821289, 'ymin': 491.9779052734, 'xmax': 218.1789855957, 'ymax': 709.9759521484, 'confidence': 0.2729455531, 'class': 73, 'name': 'book'}]
```