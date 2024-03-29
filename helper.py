import os
import pickle
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm


model = ResNet50(weights='imagenet', include_top=False, pooling='max',input_shape=(224,224,3))
model.trainable = False


filepath = []

for file in os.listdir('images'):
    filepath.append(os.path.join('images',file))



feature_vector = []

for img in tqdm(filepath):
     img = image.load_img(img, target_size=(224, 224))
     x = image.img_to_array(img)
     x = np.expand_dims(x, axis=0)
     x = preprocess_input(x)
     features = model.predict(x,verbose=0).flatten()
     normalized_features = features / norm(features)
     feature_vector.append(normalized_features)


pickle.dump(feature_vector,open('feature_vector.pkl','wb'))
pickle.dump(filepath,open('filepath.pkl','wb'))