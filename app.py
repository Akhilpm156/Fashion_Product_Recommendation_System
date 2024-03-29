import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import norm
import pickle
import streamlit as st
from PIL import Image
import os

model = ResNet50(weights='imagenet', include_top=False, pooling='max',input_shape=(224,224,3))
model.trainable = False

feature_vector_load = np.array(pickle.load(open('feature_vector.pkl','rb')))
filepath_load = pickle.load(open('filepath.pkl','rb'))


def feature_extract(img,model):
     img = image.load_img(img, target_size=(224, 224))
     x = image.img_to_array(img)
     x = np.expand_dims(x, axis=0)
     x = preprocess_input(x)
     features = model.predict(x,verbose=0).flatten()
     normalized_features = features / norm(features)
     return normalized_features

def recommend(feature,featurelist):
     NN = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
     NN.fit(feature_vector_load)
     distances,indices = NN.kneighbors([feature])
     return indices

# Streamlit app
def main():
    st.title("Product Recommendation System")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_file is not None:

        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Display uploaded image
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption='Uploaded Image')
        
        # saving images in uploads folder
        saved_img = os.path.join('uploads',uploaded_file.name)

        # images feature extracting
        img_features = feature_extract(saved_img,model)
        
        # similar images index finding
        similar_images_index = recommend(img_features,feature_vector_load)
    

        # Display similar images
        st.subheader("Similar Products:")

        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filepath_load[similar_images_index[0][0]])
        with col2:
            st.image(filepath_load[similar_images_index[0][1]])
        with col3:
            st.image(filepath_load[similar_images_index[0][2]])
        with col4:
            st.image(filepath_load[similar_images_index[0][3]])
        with col5:
            st.image(filepath_load[similar_images_index[0][4]])

if __name__ == "__main__":
    main()