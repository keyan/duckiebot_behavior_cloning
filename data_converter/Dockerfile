# Definition of data download and extraction container
# We start from a ros image
FROM ros:noetic-perception-focal

# DO NOT MODIFY
RUN apt-get update -y && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

# we make the workspace our working directory
WORKDIR /workspace
# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
COPY requirements.txt ./
RUN pip3 install -U "pip>=20.2"
RUN  pip3 install --use-feature=2020-resolver -r requirements.txt


# if you have more file use the COPY command to move them to the workspace
# Unnecessary files are ignored using .dockerignore file
COPY src/ /workspace/src

# Extract data into useable format
CMD python3 src/extract_data.py