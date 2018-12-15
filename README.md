![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

## Installation
```bash
cd yolov3
make
mkdir weight
cd weight
wget https://pjreddie.com/media/files/yolov3.weights
cd ../../
```

## Example
```python
from yolov3.yolov3 import Yolov3
import cv2

image = cv2.imread("traffic.jpg")
obj_detector = Yolov3()
detected_objects = obj_detector.detect_image("traffic.jpg")
for obj in detected_objects:
    Yolov3.draw_bboxes(image, obj)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```