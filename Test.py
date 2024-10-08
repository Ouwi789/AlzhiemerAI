import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('/Path/To/model1.keras')

def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)                     
    img_array = np.expand_dims(img_array, axis=0)          
    img_array = img_array / 255.0                              
    return img_array

test_image_path = sys.argv[1]
preprocessed_image = preprocess_image(test_image_path, target_size=(128, 128))

prediction = model.predict(preprocessed_image, verbose = 0)

#I've lowered the threshold here, usually it is 0.5
predicted_class = 1 if prediction >= 0.3 else 0

string_class = "No Dementia" if predicted_class == 1 else "Dementia"

certainty_percent = prediction * 100 if prediction >= 0.5 else (1-prediction)*100

print(f'{string_class}|{certainty_percent}')
