FROM ubuntu:18.04
LABEL maintainer="cal.loomis@gmail.com"

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-pip python3-opencv -qq && \
    apt-get install -y -qq --no-install-recommends usbutils git && \
    pip3 install requests flask && \
    apt-get clean && \
    rm -fr /var/lib/apt/lists

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN cd /root && \
    git clone https://github.com/loomis/nuvlabox-video.git

CMD ["/root/nuvlabox-video/app.py", "80", "640", "480", "4"]
