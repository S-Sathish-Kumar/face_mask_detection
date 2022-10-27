# import the opencv library
import cv2 
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('keras_model.h5')


# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
  
    #Modify the input data by
    # 1.Resizing the image
    img = cv2.resize(frame,(224,224))
    # 2.Converting the image imto Numpy array and increase dimension
    test_image = np.array(img, dtype = np.float32)
    test_image = np.expand_dims(test_image, axis = 0)
    # 3.Normalising the image
    normalised_image = test_image/255
    #predict result
    prediction = model.predict(normalised_image)
   
    print("Prediction : ",prediction)

    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        print("Closing")
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()