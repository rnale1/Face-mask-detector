#https://dev.to/whitphx/build-a-web-based-real-time-computer-vision-app-with-streamlit-57l2
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
@st.cache(allow_output_mutation=True)
def load_model1():
    model = load_model('./model/mobilenet_facemaskdetector.h5')
    return model

def facemaskdetector():
    model = load_model1()
    labels_dict = {1: 'without mask', 0: 'mask'}
    color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

    size = 4
    @st.cache(allow_output_mutation=True)
    def get_cap():
        return cv2.VideoCapture(0)

    webcam = get_cap()

    frameST = st.empty()

    # We load the xml file
    classifier = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
    while True:
        (rval, im) = webcam.read()

        im = cv2.flip(im, 1, 1)  # Flip to act as a mirror

        # Resize the image to speed up detection
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))



        # detect MultiScale / faces
        faces = classifier.detectMultiScale(mini)

        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
            # Save just the rectangle faces in SubRecFaces
            face_img = im[y:y + h, x:x + w]
            resized = cv2.resize(face_img, (224, 224))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 224, 224, 3))
            reshaped = np.vstack([reshaped])
            result = model.predict(reshaped)
            # print(result)

            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
            cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        frameST.image(im, channels="BGR")
    cv2.destroyAllWindows()

def main():

    st.title("Face Mask Detection APP")
    activities=["About","Detection"]
    choice = st.sidebar.selectbox("select activity", activities)
    if choice == 'About':
        img = Image.open("./image/img.jpeg")
        st.image(img)
        st.subheader("Face mask detector using MobileNet")
        st.markdown("Built with Streamlit by [Renuka Nale and Isaac Wagner](https://github.com/Nale123)")
        st.success("rnale1@binghamton.edu, iwagner3@binghamton.edu")
    elif choice == 'Detection':
        facemaskdetector()


if __name__ == '__main__':
		main()