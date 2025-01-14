FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN ln -fs /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.riken.go.jp/Linux/ubuntu/%g" /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y sudo git vim wget curl cmake unzip openssh-server g++ \
        libgl1-mesa-dev libglib2.0-0 libopenexr-dev sqlite3 ffmpeg

RUN useradd -m -d /home/user -G sudo -u 1000 user -s /bin/bash && \
    usermod -aG sudo user && \
    echo "user:user" | chpasswd && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -u -p /opt/miniconda

COPY environment.yaml /environment.yaml
RUN /opt/miniconda/bin/conda env create -f /environment.yaml
COPY requirements.txt /requirements.txt
RUN /opt/miniconda/bin/conda run -n braking pip install -r /requirements.txt

RUN mkdir /run/sshd && \
    echo "ChallengeResponseAuthentication no" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config && \
    echo "UsePAM no" >> /etc/ssh/sshd_config && \
    echo "PermitRootLogin no" >> /etc/ssh/sshd_config && \
    echo "AuthorizedKeysFile /home/user/.ssh_mnt/authorized_keys" >> /etc/ssh/sshd_config

WORKDIR /home/user/braking

ENTRYPOINT ["/usr/sbin/sshd", "-D"]
