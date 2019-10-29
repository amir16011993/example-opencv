#!/usr/bin/env python
# coding:utf-8

import os
import cv2
import time
import utils
import threading
import collections
import requests

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
        self.identify_fps = Fps(15)

        self._faces = []
        self._faces_lock = threading.Lock()

        self.identify_faces_queue = utils.RenewQueue()
        self.prepare_frame_queue = utils.RenewQueue()
        self.request_image_queue = utils.RenewQueue()

        self.video.set(3, width)
        self.video.set(4, height)
        self.width = int(self.video.get(3))
        self.height = int(self.video.get(4))
        print('%sx%s' % (self.width, self.height))

        self.identify_faces_threads_number = threads

        self.get_frame_thread = threading.Thread(target=self.run_get_frame, name='get_frame')
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        self.prepare_frame_thread = threading.Thread(target=self.run_prepare_frame, name='prepare_frame')
        self.prepare_frame_thread.daemon = True
        self.prepare_frame_thread.start()

        self.identify_faces_threads = [threading.Thread(target=self.run_identify_faces, name='identify_faces#%i' % (i+1,))
                                       for i in range(self.identify_faces_threads_number)]
        for thread in self.identify_faces_threads:
            thread.daemon=True
            thread.start()

    def __del__(self):
        self.video.release()

    @property
    def faces(self):
        with self._faces_lock:
            return self._faces

    @faces.setter
    def faces(self, value):
        with self._faces_lock:
            self._faces = value

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

    def run_identify_faces(self):
        while True:
            frame = self.identify_faces_queue.get()
            self.identify_faces(frame)

    def script_path(self):
        return os.path.dirname(os.path.realpath(__file__))

    #@profile
    def identify_faces(self, frame):
        cascade_path = os.path.join(self.script_path(),
                                    'haarcascade_frontalface_default.xml')
        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(cascade_path)

        # Get a grayscale verion of the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        self.faces = faceCascade.detectMultiScale(
           gray,
           scaleFactor=1.1,
           minNeighbors=20,
           minSize=(40, 40),
           flags = cv2.CASCADE_SCALE_IMAGE
        )

        self.identify_fps.new_frame()

    @staticmethod
    def send_to_influxdb(url, payload):
        try:
            requests.post(url, data=payload.encode())
        except Exception as e:
            print("Unable to write into InfluxDB: %s" % e)
            pass

    def draw_faces_rectangles(self, frame):
        ''' Draw a rectangle around the faces '''
        if not isinstance(self.faces, list) and type(self.faces).__module__ == "numpy":

            db = os.getenv("INFLUXDB_DATABASE")
            base_url = os.getenv("INFLUXDB_ENDPOINT")
            if base_url == "/":
                base_url = base_url[0:-1]

            payload = "face_area value=%.2f" % ( (self.faces[0][2] * self.faces[0][3]))

            url = '%s/write?db=%s' % (base_url, db)

            tmp_thread = threading.Thread(target=self.send_to_influxdb, name='send_to_influxdb', args=[url, payload])
            tmp_thread.daemon = True
            tmp_thread.start()



        for (x, y, w, h) in self.faces:
           cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def draw_fps(self, frame):
        camera_fps = self.camera_fps()
        if camera_fps is not None:
            cv2.putText(frame, '{:5.2f} camera fps'.format(camera_fps),
                        (10,self.height-50), self.font, 0.6, (250,25,250), 2)

        network_fps = self.network_fps()
        if network_fps is not None:
            cv2.putText(frame, '{:5.2f} effective fps'.format(network_fps),
                        (10,self.height-30), self.font, 0.6, (250,25,250), 2)

        identify_fps = self.identify_fps()
        if identify_fps is not None:
            cv2.putText(frame, '{:5.2f} identifications/sec'.format(identify_fps),
                        (10,self.height-10), self.font, 0.6, (250,25,250), 2)

    def draw_date(self, frame):
        cv2.putText(frame, time.strftime("%c"), (10,20), self.font, 0.6,
                    (250,25,250), 2)

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
        ret, jpeg = cv2.imencode('.jpg', frame,
                                 (cv2.IMWRITE_JPEG_QUALITY, self.quality))
        return jpeg.tobytes()

    #@profile
    def prepare_frame(self, frame):
        self.draw_fps(frame)
        self.draw_date(frame)
        self.draw_faces_rectangles(frame)

    #@profile
    def request_image(self):
        image = self.request_image_queue.get()
        self.network_fps.new_frame()
        return image

    # Not used. Old synchronous version
    def get_image(self):
        frame = self.get_frame()
        self.identify_faces(frame)
        self.draw_fps(frame)
        self.draw_date(frame)
        self.draw_faces_rectangles(frame)
        return self.encode_frame_to_jpeg(frame)

    def mjpeg_generator(self):
        """Video streaming generator function."""
        while True:
            image = self.request_image()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


def main():
    print(Camera().request_image())


if __name__ == "__main__":
    main()

