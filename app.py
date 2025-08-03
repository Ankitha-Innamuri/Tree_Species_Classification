!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# ========== Load Data and Models ==========

# Remove this function as tree_species_model.h5 is a Keras model, not a pickle file
# @st.cache_data
# def load_data():
#     return pd.read_pickle('tree_species_model.h5')

# Remove this function as scaler.joblib and nn_model.joblib were not created
# @st.cache_resource
# def load_nn_models():
#     scaler = joblib.load('scaler.joblib')
#     nn_model = joblib.load('nn_model.joblib')
#     return scaler, nn_model

@st.cache_resource
def load_cnn_model():
    # Load the basic CNN model which was successfully trained and saved
    return load_model("basic_cnn_tree_species.h5")

# Placeholder for a DataFrame containing tree information.
# This needs to be created or loaded from a file if available.
# For now, we'll create a dummy DataFrame with class labels from the image directories
@st.cache_data
def load_tree_info_df(repo_path):
    class_labels = sorted(os.listdir(repo_path))
    # Create a simple DataFrame with just the class names.
    # In a real application, you would load a CSV or other file with tree information.
    df = pd.DataFrame({'common_name': class_labels})
    return df


# ========== Utility Functions ==========

# Remove this function as it relies on the non-existent nn_model and scaler
# def recommend_species(input_data, nn_model, scaler, df, top_n=5):
#     input_scaled = scaler.transform([input_data])
#     distances, indices = nn_model.kneighbors(input_scaled)
#     neighbors = df.iloc[indices[0]]
#     species_counts = Counter(neighbors['common_name'])
#     top_species = species_counts.most_common(top_n)
#     return top_species

# Modify this function to work with the simplified df
def get_common_locations_for_species(df, tree_name, top_n=10):
    # Since the current df only has 'common_name', we can't provide location info.
    # In a real app, you would filter a more complete df here.
    st.warning("Location data is not available in this simplified demo.")
    return pd.DataFrame(columns=['city', 'state', 'count'])


# ========== Main App ==========

def main():
    st.title("🌿 Tree Intelligence Assistant")

    repo_path = "Tree_Species_Classification/Tree_Species_Dataset"
    # Load the simplified tree info DataFrame
    df = load_tree_info_df(repo_path)


    # Load the CNN model
    cnn_model = load_cnn_model()

    # Get class labels from the directory structure
    class_labels = sorted(os.listdir(repo_path))

    mode = st.sidebar.radio("Choose Mode", [
        "📷 Identify Tree from Image" # Removed other modes as they depend on missing data/models
    ])

    # Removed modes "🌲 Recommend Trees by Location" and "📍 Find Locations for a Tree"


    if mode == "📷 Identify Tree from Image":
        st.write("Upload a tree image to predict its species.") # Simplified description
        uploaded_file = st.file_uploader("Choose a tree image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)

            IMG_SIZE = (224, 224)
            img = image.resize(IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = cnn_model.predict(img_array)
            pred_idx = np.argmax(predictions)
            # Ensure pred_idx is within the bounds of class_labels
            if pred_idx < len(class_labels):
                pred_label = class_labels[pred_idx]
                confidence = predictions[0][pred_idx]

                st.success(f"🌳 Predicted Tree Species: **{pred_label}**")
                st.write(f"🔍 Confidence: **{confidence:.2%}**")

                # Show top-3
                st.subheader("🔝 Top 3 Predictions:")
                top_3_idx = predictions[0].argsort()[-3:][::-1]
                for i in top_3_idx:
                     if i < len(class_labels): # Ensure index is valid
                         st.write(f"{class_labels[i]} - {predictions[0][i]:.2%}")
            else:
                 st.error("Prediction index out of bounds. Could not determine species.")


            # Removed location recommendation as location data is not available
            # # Recommend locations
            # st.subheader(f"📌 Common Locations for '{pred_label}'")
            # location_info = get_common_locations_for_species(df, pred_label)
            # if location_info.empty:
            #     st.warning("This species is not found in the dataset.")
            # else:
            #     st.dataframe(location_info)

if __name__ == "__main__":
    main()