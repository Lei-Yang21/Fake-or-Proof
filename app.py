import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import torchvision.transforms.functional as TF
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import firestore, storage
import torchvision.models as models
import tempfile
import os
import concurrent.futures

# ---------------------- Model & Firestore Setup ----------------------
BUCKET_NAME = "superproof"
MODEL_PATH = "models/model.pth"  
LOCAL_MODEL_PATH = "model.pth"
labels = ["AI Generated Image", "Real Image"]


@st.cache_resource
def download_model_from_gcs():
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_PATH)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
            blob.download_to_filename(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return None
@st.cache_resource
def load_model():
    try:
        model_path = download_model_from_gcs()
        if not model_path:
            return None
        torch.serialization.add_safe_globals([models.mobilenetv2.MobileNetV2])
        model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        model.eval()
        os.unlink(model_path)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def get_firestore_client():
    try:
        return firestore.Client()
    except Exception as e:
        st.warning("Firestore not connected. Prediction history will not be stored.")
        return None

def validate_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("The file you uploaded is not a valid image. Please try a different file.")
        return None
    
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if w < 50 or h < 50 or w > 5000 or h > 5000:
        st.error("Image dimensions must be between 50×50 and 5000×5000.")
        return None

    return image

def reset_prediction():
    st.session_state.prediction_done = False
    st.session_state.idx = None
    st.session_state.show_history = False


def colored_progress(pct: float):
    # Decide color based on confidence
    if pct >= 80:
        bar_color = "#4caf50"   # green
    elif pct >= 50:
        bar_color = "#ffeb3b"   # yellow
    else:
        bar_color = "#f44336"   # red

    # HTML/CSS for the bar
    bar_html = f"""
    <div style='
        background: #e0e0e0;
        border-radius: 8px;
        padding: 2px;
        width: 100%;'>
      <div style='
        width: {pct:.1f}%;
        background: {bar_color};
        height: 24px;
        border-radius: 5px;
        text-align: center;
        color: black;
        font-weight: bold;
        line-height: 24px;
      '>{pct:.1f}%</div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)
model = load_model()
db = get_firestore_client()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---------------------- Session State ----------------------
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'idx' not in st.session_state:
    st.session_state.idx = None
if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# ---------------------- Sidebar ----------------------
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=None,on_change=reset_prediction)
predict_button = st.sidebar.button("Predict")
show_probs = st.sidebar.checkbox("Show class probabilities", value=True)
show_explanation = st.sidebar.checkbox("Show Model Explanation (GradCAM)", value=True)
toggle_history = st.sidebar.button("Toggle Prediction History")
if toggle_history:
    st.session_state.show_history = not st.session_state.show_history

st.sidebar.markdown("---")
st.sidebar.subheader("Model Metadata")
st.sidebar.markdown(f"**Model File:** {MODEL_PATH}")
st.sidebar.markdown("**Architecture:** MobileNetV2 (fine-tuned)")
st.sidebar.markdown("**Input Size:** 224x224 RGB")
st.sidebar.markdown("**Classes:** AI Generated, Real")

# ---------------------- Main Layout ----------------------
st.title("Fake or Real Image Classifier Dashboard")

# 1. Uploaded Image
if uploaded_file:
    image = validate_image(uploaded_file)
    if image is None:
        st.stop()
    st.subheader("1. Uploaded Image")
    st.image(image, use_container_width=True)
    st.write("**Image Preprocessing:** Resizing to 224×224 and normalizing.")

    # 2. Prediction
    if predict_button:
        inp = transform(image).unsqueeze(0)
        with torch.no_grad():
            out = model(inp)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, pred = torch.max(probs, dim=1)

        st.session_state.prediction_done = True
        st.session_state.probabilities = probs.cpu().numpy()[0]
        st.session_state.idx = pred.item()
        st.session_state.confidence_pct = float(conf) * 100

        if db:
            db.collection("predictions_history").add({
                "filename": uploaded_file.name,
                "prediction": labels[pred.item()],
                "confidence": st.session_state.confidence_pct,
                "timestamp": datetime.utcnow().isoformat()
            })

# 3. Display Prediction & Explanation
if st.session_state.prediction_done:
    lbl = labels[st.session_state.idx]
    pct = st.session_state.confidence_pct

    st.subheader("2. Prediction Result")
    st.write(f"**Prediction:** {lbl}")
    colored_progress(pct)

    if show_explanation:
        st.subheader("3. Model Explanation: GradCAM")
        try:
            cam_in = transform(image).unsqueeze(0).requires_grad_(True)
            target_layer = None
            for name, module in model.named_modules():
                if "features" in name:
                    target_layer = name  
            if target_layer is None:
                raise RuntimeError("No target layer found for GradCAM")

            cam_extractor = GradCAM(model, target_layer=target_layer)

            with torch.enable_grad():
                cam_out = model(cam_in)

            activation = cam_extractor(class_idx=st.session_state.idx, scores=cam_out)[0]

            for handle in cam_extractor.hook_handles:
                handle.remove()
            cam_extractor.hook_handles.clear()

            heatmap = TF.to_pil_image(activation, mode='F').resize(image.size)
            overlay = overlay_mask(image, heatmap, alpha=0.6)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(overlay, caption=f"GradCAM: {lbl}", use_container_width=True)
        except Exception as e:
            st.error(f"GradCAM failed: {e}")


# 4. Prediction History
if st.session_state.show_history:
    st.subheader("4. Prediction History")
    if db:
        docs = db.collection("predictions_history") \
                 .order_by("timestamp", direction=firestore.Query.DESCENDING) \
                 .stream()
        recs = []
        for d in docs:
            x = d.to_dict()
            recs.append({
                "Filename": x.get("filename"),
                "Prediction": x.get("prediction"),
                "Confidence": float(x.get("confidence", 0.0) or 0.0),
                "Timestamp": x.get("timestamp")
            })

        if recs:
            dfh = pd.DataFrame(recs)
            st.dataframe(dfh)
            csv = dfh.to_csv(index=False).encode("utf-8")
            st.download_button("Download History CSV", data=csv, file_name="history.csv", mime="text/csv")

            dfh["Timestamp"] = pd.to_datetime(dfh["Timestamp"])
            dfh = dfh.set_index("Timestamp")

            st.subheader("Confidence Over Time")
            st.line_chart(dfh["Confidence"])

            if st.button("Clear History"):
                for d in db.collection("predictions_history").stream():
                    d.reference.delete()

        else:
            st.info("No predictions made yet.")
    else:
        st.warning("Database not connected.")
