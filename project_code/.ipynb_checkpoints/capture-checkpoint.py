import sys
import logging
import os
import cv2
from utils import write_image, key_action, init_cam
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import time
import threading
from tensorflow.keras import backend as K

def predict_frame(image, model):
    # Define the classes
    classes = ['bottle','cup','spectacles','shoes','empty']
        
    # reverse color channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape image to (1, 224, 224, 3)
    image = np.expand_dims(image,axis=0)

    # apply pre-processing
    image = preprocess_input(image)

    """predict"""
    predictions = model.predict(image)

    # convert to numpy array
    array = np.array(predictions)

    # Check if all values are less than 0.70
    all_less_than_0_70 = np.all(array < 0.70)
    

    # if another object or empty
    if all_less_than_0_70:
        predicted_class_index = 4
    else:
        # Get the index of the highest probability
        predicted_class_index = np.argmax(predictions)
    
    # Get the label corresponding to the highest probability
    predicted_label = classes[predicted_class_index]

    return predicted_label

    

   

if __name__ == "__main__":

    # clear session
    K.clear_session()

    # load model
    model = load_model('image_classifier.h5')
    
    logging.getLogger().setLevel(logging.INFO)
   
    # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None

    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # draw a [224x224] rectangle into the frame, leave some space for the black border 
            offset = 2
            width = 224
            x = 160
            y = 120
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
            )     
            
            # get key event
            key = key_action()
            
            # make prediction 
            # extract the [224x224] rectangle out of it
            image = frame[y:y+width, x:x+width, :]
            prediction_label = predict_frame(image, model)
            logging.info(prediction_label)
            cv2.putText(frame,prediction_label,(50,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # time.sleep(1/20)
            
            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)            
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()
