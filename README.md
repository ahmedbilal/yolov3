![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

## Example
```python
import os
import cv2

from darknet.yolov3 import Yolov3

obj_detector = Yolov3()
im = cv2.imread(f"{os.getcwd()}/data/dog.jpg")
detected_objects = obj_detector.detect_image(f"{os.getcwd()}/data/dog.jpg")

for obj in detected_objects:
    Yolov3.draw_bboxes(im, obj)

cv2.imshow("Image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
```