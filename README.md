# Happiness Detector

## Table of Contents

- [Deployment](#deployment)
- [Overview](#overview)
- [Information on the Deep Learning model](#Information-on-the-Deep-Learning-model)
- [Reference and Remarks](#Reference-and-Remarks)

## Deployment
App can be deployed on Streamlit with ```streamlit run app.py```

Or you can run it with docker with ```docker build . -t happiness-detector``` followed by ```docker run -ti --rm happiness-detector```

![Cover page](gif/gif-extra-large.gif?raw=true "App")

## Overview
"Happiness Detector" is an emotion detector web APP deployed on Streamlit. 
The app can detect 3 emotions ("Happy", "Neutral", and "Sad") and is made up of two different deep learning models, 1. Face Detector and 2. Emotion Detector

## Information on the Deep Learning model
![Cover page](gif/algorithm_ex.png?raw=true "How the model works")

#### Face Detector- CAFFE Res10 300x300 SSD
Pretrained [CAFFE Res10 300x300 SSD](https://github.com/opencv/opencv/tree/master/samples/dnn) is used for face detection. It returns the X and Y coordinate of the bounding boxes to face detected.
Only detected faces with more than 50% confidence are classified with the emotion detector. 

#### Emotion Detector 
The detected faces are then classified with an Emotion Detector. 
TA pre-trained model - MobilenetV2 with pre-trained weights from Imagenet was utilized as the base model. 
The model is trained on roughly 75,000 emotion images from [MMI Facial Expression DB](https://mmifacedb.eu/).

Due to the imbalance dataset, class weights are used to give higher weightage to undersampled classes. 
Each image is preprocessed into "RGB" mode, size of 224 by 224 and into tensor arrays of shapped (224, 224, 3). 
The images are then batched prior to feeding into the model. 

Architecture of the model consists of the following:

- Input layer 
- Image preprocessing layer
- Base Model: MobileNetv2
- Global Average Pooling Layer
- Dense layer 
- Dropout layer
- Output dense layer

Model training codes can be found in [src/train.py](src/train.py)

## Reference and Remarks
- [Streamlit Webrtc](https://github.com/whitphx/streamlit-webrtc) - For webcam streaming on StreamLit
- [Face Mask Detection](https://github.com/chandrikadeb7/Face-Mask-Detection) - For model architectural 
- [Data - MMI Facial Expression Database](https://mmifacedb.eu/) - Dataset