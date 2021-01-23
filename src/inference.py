from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import imutils


def detect_and_predict_emotions(frame, faceNet, emotionsNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (160, 160),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (160, 160))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = emotionsNet.predict(faces, batch_size=32)

	return locs, preds


def return_annotated_images(frame, faceNet, emotionsNet):
	frame = imutils.resize(frame, width=400)
	(locs, preds) = detect_and_predict_emotions(frame, faceNet, emotionsNet)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		labels = {
			0: 'happy',
			1: 'neutral',
			2: 'sad'
		}
		colors = {
			0: (111, 113, 232),  # happu
			1: (255, 255, 255),  # neutral
			2: (193, 137, 101)  # sad
		}
		index = np.argmax(pred)
		emotion = labels[index]
		color = colors[index]
		prob = np.max(pred)

		label = f"{emotion}: {prob :.2f}"

		frame = cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		frame = cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	return frame
