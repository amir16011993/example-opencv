#!/usr/bin/env python
# coding: utf-8

import sys

from flask import Flask, render_template, Response
from camera import Camera  # emulated camera


app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page which makes use of /mjpeg."""
    return render_template('index.html')

@app.route('/video')
def video():
    """Video streaming home page which makes use of /jpeg."""
    return render_template('video.html')

@app.route('/mjpeg')
def mjpeg():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(Camera(**get_parameters()).mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    direct_passthrough=True)

@app.route('/jpeg')
def jpeg():
    return Response(Camera(**get_parameters()).request_image(),
                    mimetype='image/jpeg',
                    direct_passthrough=True)

def get_parameters():
    p = dict()
    nb = len(sys.argv)
    if nb >= 2: p['quality'] = min(100, max(0,int(sys.argv[1])))
    if nb >= 3: p['width']   = max(50, int(sys.argv[2]))
    if nb >= 4: p['height']  = max(50, int(sys.argv[3]))
    if nb >= 5: p['threads'] = max(0, int(sys.argv[4]))
    return p


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True)

