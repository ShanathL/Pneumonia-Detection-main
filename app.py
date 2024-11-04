import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64
import warnings
from fpdf import FPDF

warnings.simplefilter(action='ignore', category=FutureWarning)

# Custom CSS for a modern, cohesive look with a glassy effect
st.markdown("""
    <style>
    .center-text {
        text-align: center;
    }
    .glassy-box {
        margin-top: 20px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .stRadio div {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .zoom {
        transition: transform .2s ease-in-out;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2), 0 6px 20px rgba(0, 0, 0, 0.19);
    }
    .zoom:hover {
        transform: scale(1.5);
    }
    .result-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 50px;
        padding: 20px;
        border-radius: 10px;
        width: 80%;
    }
    .separator {
        width: 50%;
        height: 1px;
        background: #3498db;
        margin: 20px auto;
    }
    .icon {
        display: block;
        margin: 10px auto;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Pneumonia Detection Image Classifier")

# Glassy box for the selection and input fields
st.markdown('<div class="glassy-box center-text"><h3>Choose an option to upload a Chest X-ray Image for Pneumonia Detection</h3>', unsafe_allow_html=True)
option = st.radio("", ["Provide URL", "Upload from Local"], index=0, key="input_method")

# Display respective input method
if option == 'Upload from Local':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown('<div class="icon">📁</div>', unsafe_allow_html=True)
    path = None
else:
    path = st.text_input('Enter Image URL to Classify', 'https://raw.githubusercontent.com/mvram123/Pneumonia-Detection/main/samples/v1.jpeg')
    st.markdown('<div class="icon">🌐</div>', unsafe_allow_html=True)
    uploaded_file = None
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/model.h5')
    return model

with st.spinner('Loading Model Into Memory...'):
    model = load_model()

classes = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']

def decode_img(image):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [224, 224])
    return np.expand_dims(image, axis=0)

def get_image_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_pdf(result, confidence, img_path):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", size=12, style='B')
    pdf.set_text_color(0, 0, 128)
    pdf.cell(200, 10, txt="Pneumonia Detection Report", ln=True, align="C")

    # Image
    pdf.image(img_path, x=10, y=30, w=100)

    # Result and confidence
    pdf.set_font("Arial", size=12)
    pdf.ln(80)  # move 80 down
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True, align="L")

    # Additional Information
    pdf.ln(10)
    pdf.set_font("Arial", size=8)
    pdf.set_text_color(200, 0, 0)
    pdf.multi_cell(0, 10, 
    """Pneumonia is an infection that inflames the air sacs in one or both lungs. 
    The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing. 
    A variety of organisms, including bacteria, viruses, and fungi, can cause pneumonia.s
    Risk Factors:
    - Infants and young children
    - People older than age 65
    - People with weakened immune systems
    - Chronic disease""")

    #Please consult a healthcare provider for further assistance.
    pdf.ln(10)
    pdf.set_font("Arial", size=8)
    pdf.set_text_color(200, 0, 0)
    pdf.multi_cell(0, 10, 
    """Please consult a healthcare provider or a Pulmonologist for further assistance.""")

    # Footer
    pdf.ln(20)
    pdf.set_font("Arial", size=8)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(200, 10, txt="© 2024 Pneumonia Detection Project. | Contact: lvsahanath@gmail.com.com", ln=True, align="C")

    pdf.output("Pneumonia_Detection_Report.pdf")
    return "Pneumonia_Detection_Report.pdf"

# Function to check if the image is likely a chest X-ray
def is_chest_xray(image):
    if image.mode not in ['L', 'RGB']:  # Check if image is grayscale or RGB
        return False
    return True

# Display uploaded image or URL image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True, output_format="JPEG")
    img_to_process = uploaded_file
elif path:
    st.image(path, caption="URL Image", use_column_width=True)
    img_to_process = path
else:
    img_to_process = None

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Add a submit button
if st.button('Submit', key='submit-btn'):
    if img_to_process is not None:
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
        else:
            content = requests.get(path).content
            img = Image.open(BytesIO(content))
        
        if not is_chest_xray(img):
            st.write("The uploaded image does not appear to be a chest X-ray. Please upload a chest X-ray image.")
            st.stop()
        
        img_np = np.array(img)
    else:
        st.write("No image provided")
        st.stop()
    
    img_np = img_np / 255.0  # Normalize the image
    
    with st.spinner('Classifying...'):
        img_preprocessed = decode_img(img_np)
        label = np.argmax(model.predict(img_preprocessed), axis=1)
    prediction = classes[label[0]]
    confidence = np.max(model.predict(img_preprocessed)) * 100

    result_color = "black" if prediction != 'Normal' else "green"

    # Generate PDF after classification
    img_path = "temp.jpg"
    img.save(img_path)
    pdf_path = generate_pdf(prediction, confidence, img_path)

    # Display result with the download button inside the result container
    st.markdown(f"""
        <style>
        .result-container {{
            background: {result_color};
            color: white;
        }}
        </style>
        <div class="result-container">
            <h3>{prediction} ({confidence:.2f}% confidence)</h3>
            <div class="zoom-container">
                <img src="data:image/jpeg;base64,{get_image_base64(img)}" class="zoom" alt="Pneumonia Detection">
            </div>
            {f"<h4>Details: {prediction} detected. Please consult a healthcare provider for further assistance.</h4>" if prediction != 'Normal' else ""}
        </div>
    """, unsafe_allow_html=True)

    # Add Streamlit download button inside the result container
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(label="Download PDF Report", data=pdf_file, file_name="Pneumonia_Detection_Report.pdf", mime="application/pdf")
