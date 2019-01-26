from ctypes import *
import random
import os
import cv2


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class DetectedObject(object):
    def __init__(self, _type, _probability, _x, _y, _w, _h):
        self.type = _type
        self.probability = _probability
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h

    def xa(self):
        return self.x

    def ya(self):
        return self.y

    def xb(self):
        return self.x + self.w

    def yb(self):
        return self.y + self.h

    def bbox(self):
        return self.x, self.y, self.x + self.w, self.y + self.h


class Yolov3(object):
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    lib = CDLL(f"{REPO_ROOT}/libdarknet.so", RTLD_GLOBAL)
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    predict = lib.network_predict
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)

    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE

    get_network_boxes = lib.get_network_boxes
    get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
    get_network_boxes.restype = POINTER(DETECTION)

    make_network_boxes = lib.make_network_boxes
    make_network_boxes.argtypes = [c_void_p]
    make_network_boxes.restype = POINTER(DETECTION)

    free_detections = lib.free_detections
    free_detections.argtypes = [POINTER(DETECTION), c_int]

    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    network_predict = lib.network_predict
    network_predict.argtypes = [c_void_p, POINTER(c_float)]

    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]

    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    do_nms_obj = lib.do_nms_obj
    do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    do_nms_sort = lib.do_nms_sort
    do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    free_image = lib.free_image
    free_image.argtypes = [IMAGE]

    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE

    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]

    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)

    @staticmethod
    def classify(net, meta, _im):
        out = Yolov3.predict_image(net, _im)
        res = []
        for i in range(meta.classes):
            res.append((meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    @staticmethod
    def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
        _im = Yolov3.load_image(image, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        Yolov3.predict_image(net, _im)
        dets = Yolov3.get_network_boxes(net, _im.w, _im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if nms:
            Yolov3.do_nms_obj(dets, num, meta.classes, nms)

        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        Yolov3.free_image(_im)
        Yolov3.free_detections(dets, num)
        return res

    def __init__(self):
        parent_of_curdir = os.path.dirname(os.getcwd())

        # https://stackoverflow.com/questions/7256283/differences-in-ctypes-between-python-2-and-3
        # Strings are treated as Unicode in Python 3. Darknet's Python wrapper is written
        # with Python 2 in mind which treat string as ASCII. So, If we need to use Python 3
        # with Darknet we need to convert/encode our strings to ASCII.

        cfg_file = f"{Yolov3.REPO_ROOT}/cfg/yolov3.cfg".encode("ascii")
        weight_file = f"{Yolov3.REPO_ROOT}/weight/yolov3.weights".encode("ascii")

        self.net = Yolov3.load_net(cfg_file, weight_file, 0)
        self.meta = Yolov3.load_meta(f"{Yolov3.REPO_ROOT}/cfg/coco.data".encode("ascii"))

    def detect_image(self, input_image):
        r = Yolov3.detect(self.net, self.meta, input_image.encode("ascii"))
        detected_obj_list = []
        for detected_obj in r:
            _type = detected_obj[0]

            _cx = int(detected_obj[2][0])
            _cy = int(detected_obj[2][1])
            _w = int(detected_obj[2][2])
            _h = int(detected_obj[2][3])
            _x = int(_cx - _w / 2)
            _y = int(_cy - _h / 2)

            _obj = DetectedObject(_type=_type, _probability=detected_obj[1],
                                  _x=_x, _y=_y, _w=_w, _h=_h)

            detected_obj_list.append(_obj)
        return detected_obj_list

    @staticmethod
    def draw_bboxes(_image, detected_object):
        cv2.rectangle(_image,
                      (detected_object.xa(), detected_object.ya()),
                      (detected_object.xb(), detected_object.yb()),
                      (0, 255, 0),
                      3)


if __name__ == "__main__":
    obj_detector = Yolov3()
    im = cv2.imread(f"{Yolov3.REPO_ROOT}/data/dog.jpg")
    detected_objects = obj_detector.detect_image(f"{Yolov3.REPO_ROOT}/data/dog.jpg")

    for obj in detected_objects:
        Yolov3.draw_bboxes(im, obj)

    cv2.imshow("Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
