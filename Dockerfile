FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04
EXPOSE 5000
EXPOSE 80

# Add some dependencies
RUN apt-get update -y -qq && apt-get install -y --no-install-recommends \
 build-essential \
 bzip2 \
 curl \
 libgl1-mesa-glx \
 wget \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

#Install Anaconda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
ENV PATH /opt/conda/bin:$PATH


# Add and install our conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
COPY . /code
WORKDIR /
RUN chmod 777 -R /code
WORKDIR /code/applications/strax_worker

# Set interpreter path to our conda environment
ENV PATH /opt/conda/envs/strax-worker-env/bin:$PATH
ENV PYTHONPATH /code:$PYTHONPATH
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ENTRYPOINT ["python"]
