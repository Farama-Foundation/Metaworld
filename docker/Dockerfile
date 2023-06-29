ARG PARENT_IMAGE=ubuntu:20.04
FROM $PARENT_IMAGE

# apt dependencies
RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Dockerfile deps
    git \
    curl \
    python3 \
    python3-venv \
    python3-dev \
    # mujoco_py
    # See https://github.com/openai/mujoco-py/blob/master/Dockerfile
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    # OpenAI baselines
    libopenmpi-dev \
    # virtualenv
    libbz2-dev \
    libreadline-dev \
    libssl-dev \
    libsqlite3-dev \
    xvfb \
    pulseaudio && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  mkdir -p -m 1777 /tmp/.X11-unix

ARG user=metaworld-user
ARG uid=999
RUN groupadd -g $uid $user && \
    useradd -m -r -u $uid -g $user $user && \
    chown -R $user:$user /home/$user
ENV USER $user
USER $user
ENV HOME /home/$user
WORKDIR $HOME

RUN mkdir -p $HOME/.mujoco && \
  curl -O https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
  tar xf mujoco210-linux-x86_64.tar.gz --directory $HOME/.mujoco

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

RUN python3 -m venv $HOME/venv

# Copy setup.py first, so that the Docker cache doesn't expire until
# dependencies change
COPY --chown=$USER:$USER setup.py $HOME/code/metaworld/setup.py
WORKDIR $HOME/code/metaworld

# Install metaworld dependencies
RUN . $HOME/venv/bin/activate && exec pip install -e .[dev]

# Add code stub last
COPY --chown=$USER:$USER . $HOME/code/metaworld

ENTRYPOINT ["docker/entrypoint-headless.sh"]
