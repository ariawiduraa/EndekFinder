import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import google.generativeai as genai
import base64

# === Load Gemini API Key ===
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    GEMINI_AVAILABLE = True
except (FileNotFoundError, KeyError):
    gemini_model = None
    GEMINI_AVAILABLE = False

# === Load Model and CSV Info ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_data
def load_info():
    return pd.read_csv("info.csv")

model = load_model()
info_df = load_info()
label_names = sorted(info_df["motif_name"].tolist())

# === Predict Function ===
def predict_image(image):
    image = image.convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    preds = model.predict(img_array)[0]
    return preds

# === Gemini Answer Functions ===
def retrieve_context_from_df(question, df):
    context = ""
    for token in question.lower().split():
        matches = df[df['motif_name'].str.lower() == token]
        if not matches.empty:
            for _, row in matches.iterrows():
                context += f"- Motif: {row['motif_name']}, Origin: {row['place_origin']}, Usage: {row['usage']}\n"
            return "Based on the database, here is some relevant information:\n" + context
    return None

def ask_gemini(motif_name, info_row=None):
    if not GEMINI_AVAILABLE:
        return "❌ Gemini API not available. Please set your API Key in secrets.toml."
    prompt = f"Explain in detail about the Balinese Endek motif called '{motif_name}'."
    if info_row is not None:
        prompt += f" Here is some known information - Origin: {info_row['place_origin']}, Material: {info_row['material']}, Usage: {info_row['usage']}, Meaning: {info_row['philosophical_meaning']}"
    try:
        result = gemini_model.generate_content(prompt)
        return result.text
    except Exception as e:
        return f"Gemini error: {e}"

def ask_gemini_general(question, context=None):
    if not GEMINI_AVAILABLE:
        return "❌ Gemini API not available. Please set your API Key in secrets.toml."
    final_prompt = "You are a helpful and friendly expert on Balinese culture, specializing in Endek textiles. Answer the user's question.\n\n"
    if context:
        final_prompt += f"Use the following information from my database as the primary context for your answer:\n{context}\n\n"
    final_prompt += f"User's Question: {question}"
    try:
        response = gemini_model.generate_content(final_prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"

# === UI Setup ===
st.set_page_config(page_title="EndekFinder", layout="wide")

# === Logo + Judul ===
logo_base64 = base64.b64encode(open("logo.png", "rb").read()).decode()
st.markdown(f"""
    <div style='text-align: center; margin-top: -30px;'>
        <img src='data:image/png;base64,{logo_base64}' width='50' style='vertical-align: middle; margin-right: 10px;'/>
        <span style='font-size: 38px; vertical-align: middle; font-weight: bold;'>EndekFinder</span>
    </div>
    <style>
        .stChatInput input {{ font-size: 16px !important; }}
        .stMarkdown p {{ font-size: 16px; }}
        .stTextInput input, .stTextArea textarea {{ font-size: 16px; }}
        .custom-header {{ font-size: 28px; font-weight: 600; margin-top: 1.5rem; margin-bottom: 1rem; }}
    </style>
""", unsafe_allow_html=True)

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["📷 Classify Motif", "💬 Ask AI", "🔍 Search Motif"])

# === TAB 1: CLASSIFICATION ===
with tab1:
    st.markdown("<div class='custom-header'>📷 Classify Endek Motif</div>", unsafe_allow_html=True)
    mode = st.radio("Select input method:", ["📤 Upload Image", "📷 Use Camera"])
    image = None

    if mode == "📤 Upload Image":
        uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
    elif mode == "📷 Use Camera":
        cam = st.camera_input("Take a picture")
        if cam:
            image = Image.open(cam)

    if image:
        st.image(image, caption="Input Image", use_container_width=True)
        preds = predict_image(image)
        top_idx = np.argmax(preds)
        top_label = label_names[top_idx]
        top_conf = preds[top_idx] * 100

        if top_conf < 30:
            st.warning("⚠️ No recognizable Endek motif detected.")
        else:
            st.success(f"🎯 Detected Motif: **{top_label}** ({top_conf:.2f}%)")
            st.markdown("#### 🔍 Other possible motifs:")
            sorted_idx = np.argsort(preds)[::-1]
            for i in sorted_idx[1:4]:
                st.write(f"- {label_names[i]} ({preds[i]*100:.2f}%)")

            row_df = info_df.loc[info_df["motif_name"] == top_label]
            if not row_df.empty:
                row = row_df.iloc[0]
                st.markdown("#### 📄 Motif Info")
                st.write(f"**Origin:** {row['place_origin']}")
                st.write(f"**Material:** {row['material']}")
                st.write(f"**Usage:** {row['usage']}")
                st.write(f"**Philosophical Meaning:** {row['philosophical_meaning']}")
                if st.button(f"🔎 More about {top_label}"):
                    with st.spinner("AI is explaining..."):
                        answer = ask_gemini(top_label, row)
                        st.markdown("#### 🤖 AI Explanation")
                        st.write(answer)

# === TAB 2: CHAT TO AI ===
with tab2:
    st.markdown("<div class='custom-header'>💬 Chat with the Endek AI Assistant</div>", unsafe_allow_html=True)
    st.caption("Ask me anything about Balinese Endek cloth. This assistant is connected to your database!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I am your Endek AI assistant. How can I help you today?"}
        ]

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about Endek..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        context = retrieve_context_from_df(prompt, info_df)
        response = ask_gemini_general(prompt, context)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

# === TAB 3: SEARCH MOTIF ===
with tab3:
    st.markdown("<div class='custom-header'>🔍 Search Motif Information</div>", unsafe_allow_html=True)
    keyword = st.text_input("Enter keyword (e.g. silk, ceremony, triangle, wisdom)")
    if st.button("Search"):
        if not keyword.strip():
            st.warning("Please enter a keyword.")
        else:
            results = info_df[
                info_df.apply(lambda row: keyword.lower() in str(row).lower(), axis=1)
            ]
            if results.empty:
                st.error("No motif matches found.")
            else:
                st.success(f"Found {len(results)} motif(s):")
                for _, row in results.iterrows():
                    st.markdown("---")
                    st.subheader(f"🧵 {row['motif_name']}")
                    st.write(f"**Origin:** {row['place_origin']}")
                    st.write(f"**Material:** {row['material']}")
                    st.write(f"**Usage:** {row['usage']}")
                    st.write(f"**Philosophical Meaning:** {row['philosophical_meaning']}")

                    motif_folder_path = os.path.join("train", row["motif_name"])
                    if os.path.isdir(motif_folder_path):
                        images = [f for f in os.listdir(motif_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if images:
                            st.image(os.path.join(motif_folder_path, images[0]), caption=row["motif_name"], width=300)

                    if st.button(f"More about {row['motif_name']}", key=row['motif_name']):
                        with st.spinner("AI is explaining..."):
                            detail = ask_gemini(row['motif_name'], row)
                            st.markdown("#### 🤖 AI Explanation")
                            st.write(detail)
                    st.markdown("---")

# === FOOTER ===
st.markdown("---")
st.caption("Created by: Aria & Laura | This project is still under development.")
