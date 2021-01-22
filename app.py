import streamlit as st
import cv2
import os
from inference import return_annotated_images
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


def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def app_object_detection():
    class NNVideoTransformer(VideoTransformerBase):

        def __init__(self):
            prototxtPath = os.path.sep.join(['face_detector', "deploy.prototxt"])
            weightsPath = os.path.sep.join(['face_detector',
                                            "res10_300x300_ssd_iter_140000.caffemodel"])
            self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
            self.emotionsNet = load_model('faces.h5')

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

    # object_detection_page = "Real time object detection (sendrecv)"
    # app_mode = st.sidebar.selectbox(
    #     "Rate your happiness level today",
    #     ['1', '2', '3', '4', '5'],)
    st.subheader('Are you happy today?')

    app_object_detection()


main()