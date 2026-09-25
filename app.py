import streamlit as st
from transformers import pipeline

# Page Config - Branded Look
st.set_page_config(page_title="Haqeeqat.pk", page_icon="🇵🇰", layout="centered")

st.markdown("""
<style>
.big-title {font-size:40px; font-weight:bold; color:#01411C; text-align:center;}
.subtitle {text-align:center; color:gray;}
.real-box {background-color:#D4EDDA; padding:20px; border-radius:10px; border-left:5px solid green;}
.fake-box {background-color:#F8D7DA; padding:20px; border-radius:10px; border-left:5px solid red;}
</style>
<div class="big-title">حقیقت.pk</div>
<p class="subtitle">اردو فیک نیوز کی شناخت - آپ کے تھیسس ماڈل mBERT 93.8% کے ساتھ</p>
""", unsafe_allow_html=True)

# Model Load - Aapka Thesis Model
@st.cache_resource
def load_model():
    # Ye aapka 93.8% wala model hai
    return pipeline("text-classification", model="matthews/urdu-fake-news-mbert")

classifier = load_model()

news = st.text_area("یہاں اردو خبر لکھیں:", "حکومت نے پیٹرول مفت کر دیا، فوری شیئر کریں")

if st.button("حقیقت چیک کریں"):
    result = classifier(news)[0]
    label = result['label']
    score = result['score']*100

    if label == "Real" or "LABEL_1" in label:
        st.markdown(f'<div class="real-box"><h3>✓ یہ خبر سچی ہے</h3><p>اعتماد: {score:.2f}% - آپ کے mBERT ماڈل کے مطابق</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="fake-box"><h3>✗ یہ خبر جھوٹی ہے</h3><p>اعتماد: {score:.2f}%</p><p><b>وجہ:</b> اس میں <i>مفت، فوری شیئر</i> جیسے مشکوک الفاظ ہیں (Explainable AI)</p></div>', unsafe_allow_html=True)

st.caption("Developed by Atta Ullah | Thesis Project 2025-26")