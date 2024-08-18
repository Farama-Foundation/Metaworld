ARG PARENT_IMAGE=rlworkgroup/garage-headless
FROM $PARENT_IMAGE

# Copy setup.py first, so that the Docker cache doesn't expire until
# dependencies change
COPY --chown=$USER:$USER setup.py $HOME/code/metaworld/setup.py
WORKDIR $HOME/code/metaworld

# Install metaworld dependencies
RUN pip install -e .[dev]

# Add code stub last
COPY --chown=$USER:$USER . $HOME/code/metaworld
