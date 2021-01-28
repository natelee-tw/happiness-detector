import streamlit as st
import cv2
import os
from src.inference import return_annotated_images
from tensorflow.keras.models import load_model
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)


def app_object_detection():
    class NNVideoTransformer(VideoTransformerBase):

        def __init__(self):
            prototxtPath = os.path.sep.join(['src', 'face_detector', "deploy.prototxt"])
            weightsPath = os.path.sep.join(['src', 'face_detector',
                                            "res10_300x300_ssd_iter_140000.caffemodel"])
            self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
            modelPath = os.path.sep.join(["src", "emotions.h5"])
            self.emotionsNet = load_model(modelPath)

        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            annotated_image = return_annotated_images(image, self.faceNet, self.emotionsNet)

            return annotated_image

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=NNVideoTransformer,
        async_transform=True)

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.confidence_threshold = 0.5


def main():
    st.header("Happiness Index")
    st.subheader('Are you happy today?')
    app_object_detection()


main()