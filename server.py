from flask import Flask, render_template, request, flash, redirect, url_for
import tensorflow as tf
import numpy as np
import os
import cv2

app = Flask(__name__)
app.secret_key = "itissecret"
app.config['UPLOAD_FOLDER'] = "C:\\Users\\samis\\OneDrive\\Desktop\\Eye Disease Detection Project\\REDD2\\REDD\\static\\uploaded images"


extension_types = ["png", "jpg", "jpeg"]
file_ext = ("docx", "csv", "pdf", "xlsx", "txt")
eye_disease_list = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
model = tf.keras.models.load_model("model/efficientnetb3-Eye Disease-95.02 (1).h5")

@app.route("/")
def home():
    return render_template("index.html", img_name=" ")
@app.route("/about")
def about():
    return render_template("about.html", img_name=" ")

@app.route("/eye-prediction")
def route_to_prediction():
    return render_template('prediction.html', img_name=" ")

@app.route("/uploader", methods=["GET", "POST"])
def uploader():
    if request.method == "POST":
        path = str(request.files.get('ret-img'))
        print(path.split("'"))
        if path.split()[1] == "''":
            flash("Input Field Cannot be Empty!") 
            return redirect(url_for("uploader") + "#about")
        elif path.split("'")[1].endswith(file_ext):
            flash("Please Input An Image.")
            return redirect(url_for("uploader") + "#about")
        else:
            f = request.files['ret-img']
            img_name = f.filename
            print(img_name)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            print(img_name.split("."))
            print(f"static/uploaded images/{img_name}")

            img_path = f"static/uploaded images/{img_name}"
            input_img = cv2.imread(img_path)

            input_img_resize = cv2.resize(input_img,(224,224))
            img_reshaped = np.reshape(input_img_resize,[1,224,224,3])

            pred = model.predict(img_reshaped)
            print(pred)
            highest_acc = pred * 100
            accuracy = round(np.amax(highest_acc))
            # print(accuracy)
            pred_label = np.argmax(pred, 1)

            # print(pred_label)
            if pred_label[0] in [0, 1, 2]:
                disease = f"Eye Disease is Likely to have {eye_disease_list[pred_label[0]]}."
                return render_template("prediction.html", img_name=img_name, prediction=disease, accuracy = accuracy)
            else:
                normal = "Eye is Normal."
                return render_template("prediction.html", img_name=img_name, prediction=normal, accuracy = accuracy)
            
    else:
        return render_template("prediction.html", img_name=" ")


if __name__=="__main__":
    app.run(debug=True)
