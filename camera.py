#!/usr/bin/env python
# coding:utf-8

import os
import cv2
import time
import utils
import threading
import collections
import requests
import numpy as np
import argparse
import random
import os

#from profilehooks import profile # pip install profilehooks


class Fps(object):

    def __init__(self, buffer_size=15):
        self.last_frames_ts = collections.deque(maxlen=buffer_size)
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            len_ts = self._len_ts()
            if len_ts >= 2:
                return len_ts / (self._newest_ts() - self._oldest_ts())
            return None

    def _len_ts(self):
        return len(self.last_frames_ts)

    def _oldest_ts(self):
        return self.last_frames_ts[0]

    def _newest_ts(self):
        return self.last_frames_ts[-1]

    def new_frame(self):
        with self.lock:
            self.last_frames_ts.append(time.time())

    def get_fps(self):
        return self()


class Camera(object):
    __metaclass__ = utils.Singleton

    def __init__(self, quality=80, width=640, height=480, threads=3):
        self.quality = quality

        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)

        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.camera_fps = Fps(50)
        self.network_fps = Fps(25)
        
        self.prepare_frame_queue = utils.RenewQueue()
        self.request_image_queue = utils.RenewQueue()

        self.video.set(3, width)
        self.video.set(4, height)
        self.width = int(self.video.get(3))
        self.height = int(self.video.get(4))
        print('%sx%s' % (self.width, self.height))


        self.get_frame_thread = threading.Thread(target=self.run_get_frame, name='get_frame')
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        self.prepare_frame_thread = threading.Thread(target=self.run_prepare_frame, name='prepare_frame')
        self.prepare_frame_thread.daemon = True
        self.prepare_frame_thread.start()

    def __del__(self):
        self.video.release()
        
    def run_get_frame(self):
        while True:
            frame = self.get_frame()
            self.identify_faces_queue.put(frame)
            self.prepare_frame_queue.put(frame)

    def run_prepare_frame(self):
        while True:
            frame = self.prepare_frame_queue.get()
            self.prepare_frame(frame)
            image = self.encode_frame_to_jpeg(frame)
            self.request_image_queue.put(image)

    def script_path(self):
        return os.path.dirname(os.path.realpath(__file__))


    @staticmethod
    def send_to_influxdb(url, payload):
        try:
            requests.post(url, data=payload.encode())
        except Exception as e:
            print("Unable to write into InfluxDB: %s" % e)
            pass

    def draw_fps(self, frame):
        camera_fps = self.camera_fps()
        if camera_fps is not None:
            cv2.putText(frame, '{:5.2f} camera fps'.format(camera_fps),(10,self.height-50), self.font, 0.6, (250,25,250), 2)

        network_fps = self.network_fps()
        if network_fps is not None:
            cv2.putText(frame, '{:5.2f} effective fps'.format(network_fps),(10,self.height-30), self.font, 0.6, (250,25,250), 2)

    def draw_date(self, frame):
        cv2.putText(frame, time.strftime("%c"), (10,20), self.font, 0.6,(250,25,250), 2)

    #@profile
    def get_frame(self):
        success, frame = self.video.read()
        self.camera_fps.new_frame()
        return frame

    #@profile
    def encode_frame_to_jpeg(self, frame):
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', frame,(cv2.IMWRITE_JPEG_QUALITY, self.quality))
        return jpeg.tobytes()

    #@profile
    def prepare_frame(self, frame):
        self.draw_fps(frame)
        self.draw_date(frame)

    #@profile
    def request_image(self):
        image = self.request_image_queue.get()
        img=classify(image)
        self.network_fps.new_frame()
        return img

    def mjpeg_generator(self):
        """Video streaming generator function."""
        while True:
            image = self.request_image()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            
    def classify(image1):
    # load the COCO class labels our Mask R-CNN was trained on
        labelsPath = os.path.sep.join([mask-rcnn-coco,"object_detection_classes_coco.txt"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # load the set of colors that will be used when visualizing a given
        # instance segmentation
        colorsPath = os.path.sep.join([mask-rcnn-coco, "colors.txt"])
        COLORS = open(colorsPath).read().strip().split("\n")
        COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
        COLORS = np.array(COLORS, dtype="uint8")

        # derive the paths to the Mask R-CNN weights and model configuration
        weightsPath = os.path.sep.join([mask-rcnn-coco,"frozen_inference_graph.pb"])
        configPath = os.path.sep.join([mask-rcnn-coco,"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

        # load our Mask R-CNN trained on the COCO dataset (90 classes)
        # from disk
        print("[INFO] loading Mask R-CNN from disk...")
        net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

        # load our input image and grab its spatial dimensions
        image = cv2.imread(image1)
        (H, W) = image.shape[:2]

        # construct a blob from the input image and then perform a forward
        # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
        # of the objects in the image along with (2) the pixel-wise segmentation
        # for each specific object
        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
        end = time.time()

        # show timing information and volume information on Mask R-CNN
        print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
        print("[INFO] boxes shape: {}".format(boxes.shape))
        print("[INFO] masks shape: {}".format(masks.shape))

        # loop over the number of detected objects
        for i in range(0, boxes.shape[2]):
            # extract the class ID of the detection along with the confidence
            # (i.e., probability) associated with the prediction
            classID = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]

            # filter out weak predictions by ensuring the detected probability
            # is greater than the minimum probability
            if confidence > 0.5:
                # clone our original image so we can draw on it
                clone = image.copy()

                # scale the bounding box coordinates back relative to the
                # size of the image and then compute the width and the height
                # of the bounding box
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                boxW = endX - startX
                boxH = endY - startY

                # extract the pixel-wise segmentation for the object, resize
                # the mask such that it's the same dimensions of the bounding
                # box, and then finally threshold to create a *binary* mask
                mask = masks[i, classID]
                mask = cv2.resize(mask, (boxW, boxH),
                    interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.3)

                # extract the ROI of the image
                roi = clone[startY:endY, startX:endX]

                # now, extract *only* the masked region of the ROI by passing
                # in the boolean mask array as our slice condition
                roi = roi[mask]

                # randomly select a color that will be used to visualize this
                # particular instance segmentation then create a transparent
                # overlay by blending the randomly selected color with the ROI
                color = random.choice(COLORS)
                blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

                # store the blended ROI in the original image
                clone[startY:endY, startX:endX][mask] = blended

                # draw the bounding box of the instance on the image
                color = [int(c) for c in color]
                cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

                # draw the predicted label and associated probability of the
                # instance segmentation on the image
                text = "{}: {:.4f}".format(LABELS[classID], confidence)
                cv2.putText(clone, text, (startX, startY - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                return clone
                                


def main():
    print(Camera().request_image())


if __name__ == "__main__":
    main()

