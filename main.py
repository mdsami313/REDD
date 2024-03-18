import tensorflow as tf
import numpy as np
import os
import cv2

eye_disease_list = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
model = tf.keras.models.load_model("model/efficientnetb3-Eye Disease-95.02 (1).h5")

# path of image
img_path = f"static\\uploaded images\\dr.jpeg"

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
    print(disease)
else:
    normal = "Eye is Normal."
    print(normal)
