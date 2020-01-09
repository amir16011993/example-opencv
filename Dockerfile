FROM ubuntu:18.04
LABEL maintainer="cal.loomis@gmail.com"

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-pip python3-opencv -qq && \
    apt-get install numpy && \
    apt-get install -y -qq --no-install-recommends usbutils git && \
    pip3 install requests flask && \
    apt-get clean && \
    mkdir ~/mask-rcnn-coco && \
    wget http://www.mediafire.com/file/angdg0lb3t0urm1/colors.txt/file -P ~/mask-rcnn-coco && \
    wget http://www.mediafire.com/file/jljhgqorn2la0yk/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt/file  -P ~/mask-rcnn-coco && \
    wget http://www.mediafire.com/file/kkqcynf37zmpl2b/object_detection_classes_coco.txt/file -P ~/mask-rcnn-coco && \
    wget http://www.mediafire.com/file/6ck7uiasklxz3yc/frozen_inference_graph.pb/file -P ~/mask-rcnn-coco && \
    rm -fr /var/lib/apt/lists

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

COPY . /root/example-opencv

CMD ["/root/example-opencv/app.py", "80", "640", "480", "4"]

