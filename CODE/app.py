from flask import Flask,render_template,request,flash,send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from gtts import gTTS
import os
app=Flask(__name__)

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
          "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/services')
def services():
    return render_template('services.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/upload", methods=["POST","GET"])
def upload():
    if request.method=='POST':
        myfile=request.files['file']
        fn=myfile.filename
        mypath=os.path.join('images/', fn)
        myfile.save(mypath)
        print(fn)
        print(type(fn))
        accepted_formated=['jpg','png','jpeg','jfif','tif']
        if fn.split('.')[-1] not in accepted_formated:
            flash("Image formats only Accepted","Danger")
        new_model = load_model("model/FinalModel.h5")
        # mypath="dataset/train/b/hand1_b_bot_seg_3_cropped.jpeg"
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        prediction = np.argmax(result)

        prediction=classes[prediction]
        m="System Predicted Sign is "
        m=m+"'"+prediction+"'"

        mytext = prediction
        audio = gTTS(text=mytext, lang="en", slow=False)
        audio.save("example.mp3")
        os.system("start example.mp3")


    return render_template("upload.html", image_name=fn, text=m)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)
@app.route('/live',methods=['POST','GET'])
def live():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

    offset = 20
    imgSize = 300

    # folder = "Data/Z"
    # counter = 0

    lables = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
              "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgwhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                # imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgwhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgwhite, draw=False)
                print(prediction, index)



            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                # imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgwhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgwhite, draw=False)

                # label =classifier.getPrediction(imgwhite, draw=False)
                # # label=np.array([[label]])
                # # label=label.shape
                # print(label)
                #
                # prediction = classes[np.array(label)]
                #
                #
                # mytext = prediction
                # audio = gTTS(text=mytext, lang="en", slow=False)
                # audio.save("example.mp3")
                # os.system("start example.mp3")

            cv2.putText(imgOutput, lables[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("Imagewhite", imgwhite)

        cv2.imshow("Image", imgOutput)
        key = cv2.waitKey(1)



    return render_template('contact.html')

if __name__=='__main__':
    app.run(debug=True)