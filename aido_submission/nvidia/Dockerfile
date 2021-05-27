FROM frank1chude1qian/dt-machine-learning-base-environment:daffy-amd64

ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# Setup any additional pip packages
RUN pip3 install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN pip3 install --use-feature=2020-resolver -r .requirements.txt

# Copy all solution files to workspace
WORKDIR /submission
COPY solution.py /submission
COPY saved_nvidia_model_ang.pth /submission
COPY saved_nvidia_model_lin.pth /submission
COPY helpers.py /submission
COPY nvidia_model.py /submission

ENTRYPOINT ["python3", "solution.py"]
