"""Streamlit UI with private history and an explicit entry point."""

import logging
import time

import streamlit as st

from fake_or_proof.config import CLASS_LABELS, Settings
from fake_or_proof.history import add_prediction, clear_history, export_history
from fake_or_proof.images import InvalidImage, read_image
from fake_or_proof.model import InferenceBusy, InferenceService

logger = logging.getLogger(__name__)


@st.cache_resource(max_entries=1, ttl=3600)
def get_inference_service(settings):
    return InferenceService.load(settings)


def reset_prediction():
    st.session_state.pop("prediction", None)


def main():
    st.set_page_config(page_title="Fake or Proof", page_icon="🔎")
    st.title("Fake or Real Image Classifier")
    st.caption(
        "Images are processed in memory. Your prediction history belongs to "
        "this session and is not saved to a database."
    )
    try:
        settings = Settings.from_env()
        service = get_inference_service(settings)
    except Exception:
        logger.exception("Model initialization failed")
        st.error(
            "The classifier is temporarily unavailable. "
            "Please try again later."
        )
        service = None
        settings = None

    uploaded_file = st.sidebar.file_uploader(
        "Upload a JPEG or PNG (5 MB maximum)",
        type=["jpg", "jpeg", "png"],
        on_change=reset_prediction,
    )
    predict_clicked = st.sidebar.button(
        "Predict", disabled=service is None or uploaded_file is None
    )
    show_probabilities = st.sidebar.checkbox(
        "Show class probabilities", value=True
    )
    st.sidebar.caption("MobileNetV2 · 224×224 RGB · CPU inference")

    if uploaded_file is not None and service is not None:
        try:
            image = read_image(uploaded_file, settings)
        except InvalidImage as error:
            reset_prediction()
            st.error(str(error))
            image = None
        if image is not None:
            st.image(image, caption="Uploaded image", use_container_width=True)
            if predict_clicked:
                now = time.monotonic()
                previous = st.session_state.get(
                    "last_prediction_at", float("-inf")
                )
                if now - previous < settings.prediction_interval_seconds:
                    st.warning(
                        "Please wait a moment before the next prediction."
                    )
                else:
                    st.session_state.last_prediction_at = now
                    try:
                        class_index, probabilities = service.predict(image)
                        st.session_state.prediction = (
                            class_index,
                            probabilities,
                        )
                        add_prediction(
                            st.session_state,
                            CLASS_LABELS[class_index],
                            probabilities[class_index],
                            settings.history_limit,
                        )
                    except InferenceBusy as error:
                        st.info(str(error))
                    except Exception:
                        logger.exception("Prediction failed")
                        reset_prediction()
                        st.error(
                            "The prediction could not be completed. "
                            "Please try again later."
                        )
            prediction = st.session_state.get("prediction")
            if prediction is not None:
                class_index, probabilities = prediction
                st.subheader(CLASS_LABELS[class_index])
                st.progress(
                    float(probabilities[class_index]),
                    text=f"Model confidence: {probabilities[class_index]:.1%}",
                )
                if show_probabilities:
                    st.bar_chart(dict(zip(CLASS_LABELS, probabilities)))
                if st.button("Explain this prediction (GradCAM)"):
                    now = time.monotonic()
                    previous = st.session_state.get(
                        "last_explanation_at", float("-inf")
                    )
                    if now - previous < settings.prediction_interval_seconds:
                        st.info(
                            "Please wait a moment before requesting another "
                            "explanation."
                        )
                    else:
                        st.session_state.last_explanation_at = now
                        try:
                            st.image(
                                service.explain(image, class_index),
                                caption="GradCAM explanation",
                                use_container_width=True,
                            )
                        except InferenceBusy as error:
                            st.info(str(error))
                        except Exception:
                            logger.exception("GradCAM failed")
                            st.error(
                                "The explanation is temporarily unavailable."
                            )

    with st.expander("Your session history (last 100 predictions)"):
        history = st.session_state.get("prediction_history", [])
        if history:
            st.dataframe(history, hide_index=True)
            st.download_button(
                "Download your history (JSON)",
                export_history(st.session_state),
                file_name="prediction-history.json",
                mime="application/json",
            )
            if st.button("Clear your session history"):
                clear_history(st.session_state)
                st.rerun()
        else:
            st.info("No predictions in this session.")


if __name__ == "__main__":
    main()
