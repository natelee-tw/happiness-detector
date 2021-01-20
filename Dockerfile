# change base image as required
FROM nvidia/cuda:9.0-runtime
ENV PYTHONIOENCODING "utf-8"

ARG MINI_CONDA_SH="Miniconda3-latest-Linux-x86_64.sh"

RUN apt update && \
    apt -y install bzip2 curl gcc git vim && \
    apt-get clean

ARG USER="aisg"
ARG WORK_DIR="/home/$USER"

RUN groupadd -g 2222 $USER && useradd -u 2222 -g 2222 -m $USER && \
    chown -R 2222:2222 $WORK_DIR

USER $USER
WORKDIR $WORK_DIR
ARG CONDA_PATH="$WORK_DIR/miniconda3/bin"

RUN curl -O https://repo.anaconda.com/miniconda/$MINI_CONDA_SH && \
    chmod +x $MINI_CONDA_SH && \
    ./Miniconda3-latest-Linux-x86_64.sh -b && \
    rm $MINI_CONDA_SH

ENV PATH $CONDA_PATH:$PATH

# New dependencies not found in your base image should go in conda.yml
COPY conda.yml conda.yml
RUN conda env update -f conda.yml -n base

COPY src src
COPY scripts/start_app.sh .

CMD ["./start_app.sh"]