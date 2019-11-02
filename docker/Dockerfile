ARG PARENT_IMAGE=rlworkgroup/garage-headless
FROM $PARENT_IMAGE

# Copy setup.py first, so that the Docker cache doesn't expire until
# dependencies change
COPY setup.py /root/code/metaworld/setup.py
WORKDIR /root/code/metaworld

# Install metaworld dependencies
RUN pip install -e .[dev]

# Add code stub last
COPY . /root/code/metaworld
