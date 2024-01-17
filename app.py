from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from test import test_model
from PIL import Image
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from keras.models import model_from_json
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','mp4'}
global GlobalResult
GlobalResult = ""

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')  # Use Image module from Pillow
    img = img.resize((48, 48))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # Load and preprocess the image
    return img

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# model_path = "model.h5"  # Chemin vers le modèle
# model = load_model(model_path)

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    if request.method == 'POST':
        
        
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'})

            file = request.files['file']

            if file.filename == '':
                return jsonify({'error': 'No selected file'})

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                print("nofile")
                # Appeler la fonction test_model avec le fichier téléchargé
                image_data = preprocess_image(file_path)
                emotion_prediction = emotion_model.predict(image_data)
                maxindex = int(np.argmax(emotion_prediction))
                result= emotion_dict[maxindex]
                # result = test_model(file_path)
                # print(result)
                print("the predicted result is" ,result)
                # GlobalResult = result
                
                return jsonify({'result': result})

            return jsonify({'error': 'Invalid file format'})
        
            
    else:
        return render_template('symptoms.html')
@app.route('/')
def index():
    return render_template('index.html', title='Welcome to Health diagnosis')

@app.route('/video',methods=['GET', 'POST'])
def VideoRecognition():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            # Save the uploaded video file
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            cap = cv2.VideoCapture(video_path)

            while True:
                # Find haar cascade to draw bounding box around face
                ret, frame = cap.read()
                frame = cv2.resize(frame, (940, 540))
                if not ret:
                    break
                face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect faces available on camera
                num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

                # take each face available on the camera and Preprocess it
                for (x, y, w, h) in num_faces:
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                    roi_gray_frame = gray_frame[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                    # predict the emotions
                    emotion_prediction = emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(emotion_prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('Emotion Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            return render_template('video.html')
    else:
        return render_template('video.html')
    
@app.route('/result')
def result():
        return render_template('result.html')

@app.route('/realtime',methods=['GET', 'POST'])
def Realtime():
    if request.method == 'POST':
        cap = cv2.VideoCapture(0)

        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            if not ret:
                break
            face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces available on camera
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            # take each face available on the camera and Preprocess it
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return render_template('prevention.html')
    else:
        return render_template('prevention.html')


# @app.route('/symptoms')
# def symptoms():
#     return render_template('symptoms.html')

    

if __name__ == '__main__':
    app.run(debug=True)

