from flask import Flask, request, jsonify
import werkzeug
import os





from werkzeug.utils import secure_filename

from app import fonksiyon
from textRecognition import textRecognition
from videoDetection import videoDetect



app = Flask(__name__)

@app.route('/uploadimage', methods = ["POST"])
def uploadimage():
    if(request.method=="POST"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploadedimages" + filename)
        objectsDetected = fonksiyon("uploadedimages"+filename)
        os.remove("./uploadedimages"+filename)
        return objectsDetected



@app.route('/uploadtext', methods = ["POST"])
def uploadtext():
    if (request.method == "POST"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploadedimages" + filename)
        textDetected = textRecognition("uploadedimages" + filename)
        os.remove("./uploadedimages"+filename)
        return textDetected


@app.route('/uploadvideo', methods = ["POST"])
def uploadvideo():
    if (request.method == "POST"):
        print(request)
        video = request.files['image']
        filename = werkzeug.utils.secure_filename(video.filename)
        video.save("./uploadedvideos" + filename)
        objectsFromVideo = videoDetect("uploadedvideos"+ filename)
        os.remove("./uploadedvideos"+filename)
        return objectsFromVideo



if(__name__  =="__main__"):
    app.run(debug=True, port=57948, host='0.0.0.0')    # ipV4:57948/.............


