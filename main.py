import streamlit as st
import cv2
from PIL import Image
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import pyttsx3
from spellchecker import SpellChecker

st.set_page_config(
        page_title="Sign Language Detector",
        page_icon=" ")

st.markdown("""
        <style>
               .css-1y4p8pa {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
                .css-1544g2n{
                    padding: 0rem 1rem 1.5rem
                }
                .eqr7zpz3{
                    left: calc(-6rem);
                    width: calc(100% + 8.5rem + 0.5rem);
                }
                #predicted-texts{
                    padding: 0.5rem 0px 0rem;
                    margin-left: calc(3rem);
                }
        </style>
        """, unsafe_allow_html=True)

model = tf.keras.models.load_model(r"sign_language_detector_e6_V7.h5")
word = []
sens = []
show = False


def speak(audio):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty("rate", 200)
    engine.setProperty('voice', voices[1].id)
    engine.say(audio)
    engine.runAndWait()

def correct_spelling(input_word):
    spell = SpellChecker()
    corrected_word = spell.correction(input_word)
    return corrected_word

def check_intersection(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    if x1_1 < x2_2 and x2_1 > x1_2 and y1_1 < y2_2 and y2_1 > y1_2:
        return True  # The bounding boxes intersect
    else:
        return False  # The bounding boxes do not intersect
    
def merge_bounding_boxes(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    # Calculate the intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    # Check if there is an intersection
    if x1_inter < x2_inter and y1_inter < y2_inter:
        # Calculate the merged bounding box
        x1_merged = min(x1_1, x1_2)
        y1_merged = min(y1_1, y1_2)
        x2_merged = max(x2_1, x2_2)
        y2_merged = max(y2_1, y2_2)
        return x1_merged, y1_merged, x2_merged, y2_merged
    else:
        # No intersection, return None or raise an exception
        return None

def crop_image(image, bbox):
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

def main(video_capture=None):
    global show
    
    st.title("Indian Sign Language Recognition")
    st.sidebar.title('IMAGE REFERENCE')
    # Set up the sidebar with image and prediction buttons
    image = Image.open(r"WhatsApp Image 2023-07-27 at 3.02.01 PM.jpeg")
    st.sidebar.image(image) 

    # Use caching to maintain state across button clicks
    def open_camera():
        st.session_state.button = not st.session_state.button
    if 'button' not in st.session_state:
        st.session_state.button = False
        st.sidebar.button("Open Camera", on_click=open_camera, key='2')


    video_placeholder = st.empty()
    st.subheader("      Predicted Texts: ")
    text_container = st.empty()
    # Set up the main page layout
    col1, col2 = st.columns([3, 1])  # Adjust the ratio as needed
    # Camera capture
    with col1:
        video_capture = None  # Initialize the camera object
    
    if st.session_state.button:
        # The message and nested widget will remain on the page
        st.sidebar.button("Close Camera", on_click=open_camera, key='0')
        video_capture = cv2.VideoCapture(0)
        _, frame = video_capture.read()
        h, w, c = frame.shape
        start_time = time.time()
        frame_counter=0
        # Check if the camera is opened successfully
        if not video_capture.isOpened():
            st.error("Error: Unable to access the camera.")
            return
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Error: Unable to retrieve frame from the camera.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
                
            # Display the frame on the Streamlit app
            display = cv2.resize(frame, (550,400))
            video_placeholder.image(display, channels="RGB")

            frame_counter += 1
            # Skip frame if not divisible by 20
            if frame_counter % 20 != 0:
                continue
            result = hands.process(frame)

            hand_landmarks = result.multi_hand_landmarks
            bounding_boxes = []

            if hand_landmarks:
                show = True
                for hand_landmark in hand_landmarks:
                    x = [landmark.x for landmark in hand_landmark.landmark]
                    y = [landmark.y for landmark in hand_landmark.landmark]

                    center = np.array([np.mean(x) * w, np.mean(y) * h]).astype('int32')
                    bounding_box = (center[0] - 180, center[1] - 180, center[0] + 180, center[1] + 180)
                    bounding_boxes.append(bounding_box)
                    cv2.circle(frame, tuple(center), 10, (255, 0, 0), 1)  # for checking the center

                if (len(bounding_boxes) > 1 and check_intersection(bounding_boxes[0], bounding_boxes[1])):
                    x1, y1, x2, y2 = merge_bounding_boxes(bounding_boxes[0],bounding_boxes[1])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cropped_image = crop_image(frame, [x1, y1, x2, y2])
                else:
                    cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 2)
                    cropped_image = crop_image(frame, bounding_box)

                display = cv2.resize(frame, (550,400))
                video_placeholder.image(display, channels="RGB")
                if cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
                    continue
                cropped_image = cv2.resize(cropped_image, (180, 180))
                cropped_image = np.expand_dims(cropped_image, 0)

                # prediction one th face image
                out = model.predict(cropped_image)
                idx = np.argmax(out)

                if idx > 9:
                    label = chr(idx - 10 + 65)
                else:
                    label = idx

                acc = np.max(out[0])
                acc=round(acc*100,2)
                if(acc > 75):
                    word.append(str(label))
                    speak(str(label))
            else:
                if show:
                    if "".join(word).isalpha():
                        target = correct_spelling("".join(word))
                    else:
                        target = "".join(word)
                    speak(target)
                    if target != None:
                        sens.append(target)
                        text_container.empty()
                        text_container.write(" ".join(sens).lower())
                    word.clear()
                    show = False
        
        # Release the camera and destroy all OpenCV windows when prediction stops
        if video_capture:
            video_capture.release()

    else:
        st.sidebar.button("Open Camera", on_click=open_camera, key='2')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        # pass