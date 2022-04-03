from flask import Flask,render_template,Response
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow import keras

app=Flask(__name__)
camera=cv2.VideoCapture(0)

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

modelF= keras.models.load_model('recognition.h5')

DATA_PATH = os.path.join('MP_Gestures_all') 
actions = np.array(['hello', 'thanks', 'i love you', 'stop', 'please', 'walk', 'argue', 'yes', 'see', 'good'])
# 40 videos
no_sequences = 40
# 30 frames
sequence_length = 30

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                 
    results = model.process(image)                
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 

def draw_styled_landmarks(image, results):

    # left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    key1 = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    key2 = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([key1, key2, lh, rh])

colors = [(245,117,16), (117,245,16), (16,117,245),(16,17,245),(16,117,24),(17,25,160),(11,45,116),(170,205,165), (224, 32, 28), (22,142,100)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


## GUI


def generate_frames():
        sequence = []
        sentence = []
        threshold = 0.4
    
        while True:
                
            ## read the camera frame
            success,frame=camera.read()
            if not success:
                break
            else:
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    image, results = mediapipe_detection(frame,holistic)
                    draw_styled_landmarks(image, results)
                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)

                    sequence.append(keypoints)
                    sequence = sequence[-30:]
                    
                    if len(sequence) == 30:
                        res = modelF.predict(np.expand_dims(sequence, axis=0))[0]
                        print(actions[np.argmax(res)])
                        
                        
                    #3. Viz logic
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5: 
                            sentence = sentence[-5:]

                            # Viz probabilities
                            image = prob_viz(res, actions, image, colors)
                    
                    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    ret,buffer=cv2.imencode('.jpg',image)
                    image=buffer.tobytes()
               
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run()