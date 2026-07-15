import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration (This builds the HTML head/metadata tab)
st.set_page_config(
    page_title="AgriShield AI",
    page_icon="🌱",
    layout="centered"
)

# 2. Cache the model loading so the site stays fast and doesn't reload the file on every click
@st.cache_resource
def load_my_ai_model():
    return tf.keras.models.load_model('crop_disease_model.keras')

model = load_my_ai_model()

# 3. Create a perfect list matching the 38 trained categories exactly in order
CLASS_NAMES = [
    'Apple Scab', 'Apple Black Rot', 'Cedar Apple Rust', 'Healthy Apple', 'Healthy Blueberry',
    'Cherry Powdery Mildew', 'Healthy Cherry', 'Corn Gray Leaf Spot', 'Corn Common Rust', 
    'Corn Northern Leaf Blight', 'Healthy Corn', 'Grape Black Rot', 'Grape Black Measles', 
    'Grape Leaf Blight', 'Healthy Grape', 'Orange Citrus Greening', 'Peach Bacterial Spot', 
    'Healthy Peach', 'Pepper Bell Bacterial Spot', 'Healthy Pepper Bell', 'Potato Early Blight', 
    'Potato Late Blight', 'Potato Healthy', 'Healthy Raspberry', 'Healthy Soybean', 
    'Squash Powdery Mildew', 'Strawberry Leaf Scorch', 'Healthy Strawberry', 'Tomato Bacterial Spot', 
    'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 
    'Tomato Two-Spotted Spider Mite', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 
    'Tomato Mosaic Virus', 'Healthy Tomato'
]

# 4. Building the UI Header (standard <h1> and <p> elements)
st.title("🌱 AgriShield AI Detection System")
st.write("Upload a clear photo of a crop leaf to diagnose plant health conditions instantly.")

# 5. File Uploader Component (Renders a drag-and-drop file interface)
uploaded_file = st.file_uploader("Drop your leaf image file here...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image file uploaded by the user
    image = Image.open(uploaded_file)
    
    # Render the image container on the web page
    st.image(image, caption='Uploaded Sample Leaf Picture', use_container_width=True)
    
    with st.spinner("Analyzing image features against AI database..."):
        # 6. Preprocessing matching your exact model configuration
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        
        # If image has an Alpha transparency channel (RGBA), strip it to 3 channels (RGB)
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
            
        img_array = np.expand_dims(img_array, axis=0) # Reshapes array from (224,224,3) to (1,224,224,3)
        
        # 7. Execute Prediction Logic
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence_score = float(np.max(predictions[0])) * 100
        
        predicted_label = CLASS_NAMES[predicted_class_index]
        
    # 8. Display Results Dashboard
    st.subheader("📊 Diagnostic Report Summary")
    
    if "Healthy" in predicted_label:
        st.success(f"**Status:** {predicted_label} (Confidence: {confidence_score:.1f}%)")
        st.balloons() # Triggers fun celebration animation for healthy crops
    else:
        st.error(f"**Anomaly Detected:** {predicted_label} (Confidence: {confidence_score:.1f}%)")
        
        # Treatment Advice Dictionary
        st.markdown("### 🛠️ Recommended Remediation Actions:")
        st.write("- Isolate infected plants promptly to prevent canopy transmission.")
        st.write("- Prune damaged leaves using sterilized cutting tools.")
        st.write("- Introduce optimized targeted organic or chemical treatments if required.")
