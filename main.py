import streamlit as st

from PIL import Image
import numpy as np
import torch
# Load YOLOv5 model (make sure the path is correct and model is available)
@st.cache_resource  # caches the model across Streamlit reruns
def load_model():
    model_path = os.path.join(os.getcwd(), 'best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
    return model

model = load_model()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Leaf Health Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Leaf Health Detection"])

# Home Page
if app_mode == "Home":
    st.header("LEAF HEALTH DETECTION")
    st.image("pexels-pixabay-86397.jpg", use_column_width=True)
    st.markdown("""
        # Leaf Health Detection

        ## Introduction
        This is a deep learning-based tool to detect leaf health using a YOLOv5 model.

        ## Features
        - **Real-time detection**
        - **YOLOv5-based bounding box prediction**
        - **Streamlit UI**

        ## Technologies
        - Python
        - YOLOv5
        - PyTorch
        - Streamlit
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
        # About Dataset  

        **Acknowledgements**  
        Dataset obtained from [Roboflow/Plant-Disease-Detection](https://public.roboflow.com/).

        **Use Case**  
        Designed for binary classification and localization of healthy and unhealthy leaves using YOLOv5.
    """)

# Leaf Health Detection Page
elif app_mode == "Leaf Health Detection":
    st.header("Leaf Health Detection")
    
    uploaded_file = st.file_uploader("Upload a leaf image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Running YOLOv5 model..."):
                results = model(image)
                results.render()  # draws boxes on image

                # Show the output image with bounding boxes
                st.image(results.ims[0], caption="Prediction", use_column_width=True)

                # Display detected classes
                detected = results.pandas().xyxy[0]['name'].tolist()
                if detected:
                    st.success("Detected: " + ", ".join(set(detected)))
                else:
                    st.warning("No leaf detected, or unable to classify.")
