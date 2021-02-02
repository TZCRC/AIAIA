#https://console.cloud.google.com/gcr/images/deeplearning-platform-release/GLOBAL/tf2-gpu.2-1
#latest tag v17:gpu
FROM gcr.io/deeplearning-platform-release/tf2-gpu:m51

RUN mkdir /ml
RUN mkdir /ml/models
RUN mkdir /ml/data

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

# Get imagenet weights for keras
RUN mkdir /root/.keras
RUN mkdir /root/.keras/models

COPY aiaia_classifier/model.py /opt/model.py
COPY aiaia_classifier/utils_readtfrecords.py  /opt/utils_readtfrecords.py
COPY aiaia_classifier/utils_train.py /opt/utils_train.py
COPY aiaia_classifier/utils_loss.py /opt/utils_loss.py

RUN chmod +x /opt/model.py
