# library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#loading model
model = load_model('plant_disease_CNN/plant_disease.h5')

CLASS_NAMES = ['Corn-common_rust','Potato-Early_blight', 'Tomato-Bacterial_spot']

# setting title of app
st.title('Plant Disease Detection')
st.markdown('Upload an image of the plant leaf')

# uploading the dog image 
plant_image = st.file_uploader('Choose an image...', type='jpg')
submit = st.button('Predict')

# On predict button click 
if submit:


    if plant_image is not None:



        # Convert the file to an open cv image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
    opencv_image =  cv2.imdecode(file_bytes, 1)



    # Displaying the image
    st.image(opencv_image, channels='BGR')
    st.write(opencv_image.shape)
    # Resizing the image
    opencv_image = cv2.resize(opencv_image, (256,256))
    #Convert image to 4 dimension
    opencv_image.shape = (1,256,256,3)
    # make prediction
    Y_pred = model.predict(opencv_image)
    result = CLASS_NAMES[np.argmax(Y_pred)]
    st.title(str('This is '+result.split('-')[0]+' leaf with ' + result.split('-')[1]))
