import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Streamlit پیج کی ترتیب
st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 Bilingual Fake News Detector")
st.write("Urdu and English News Verification System")

# فائن ٹیونڈ ماڈل لوڈ کرنے کا فنکشن
@st.cache_resource
def load_model():
    # یہاں اپنے فائن ٹیونڈ ماڈل کا فولڈر یا Hugging Face کا پاتھ دیں
    MODEL_PATH = "path/to/your/fine-tuned/model" 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# یوزر ٹیکسٹ ان پٹ
user_input = st.text_area("Enter News Article / متن یہاں درج کریں:", height=150)

if st.button("Check News / تصدیق کریں"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            probabilities = F.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            
            # پروببلٹی نکالیں
            fake_prob = probabilities[0][0].item() * 100
            real_prob = probabilities[0][1].item() * 100

            if prediction == 0:
                st.error(f"🚨 *Fake News Detected!* (Confidence: {fake_prob:.2f}%)")
            else:
                st.success(f"✅ *Real News!* (Confidence: {real_prob:.2f}%)")
