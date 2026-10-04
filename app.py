import streamlit as st
from transformers import pipeline
import re

st.set_page_config(page_title="AI Fake News Detector - Atta Ullah", page_icon="📰", layout="centered")

st.sidebar.title("Thesis - Final AI Model")
st.sidebar.markdown("""
**Student:** Atta Ullah (2022-UoB-214)
**Model:** XLM-RoBERTa NLI (Real AI)
**Dataset:** Ax-to-Grind 10083
**Supervisor:** Dr. Hamid Hussain
**Accuracy:** 93.8% (mBERT)
""")

# یہ ہے اصلی AI ماڈل جو سوچ کر فیصلہ کرتا ہے
@st.cache_resource
def load_real_ai():
    # یہ multilingual ہے، اردو اور انگریزی دونوں سمجھتا ہے
    classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
    return classifier

with st.spinner("اصلی AI ماڈل لوڈ ہو رہا ہے... پہلی بار 1-2 منٹ لگے گا"):
    ai_classifier = load_real_ai()

st.title("📰 اصلی AI سے جعلی خبر کی شناخت")
st.markdown("یہ سسٹم اب کی-ورڈ نہیں، بلکہ AI کی سوچ سے فیصلہ کرے گا")
st.info("یہ XLM-RoBERTa ماڈل ہے جو آپ کے تھیسس کے mBERT 93.8% کی طرح ہی multilingual ہے")

news_text = st.text_area("خبر لکھیں / Enter News:", height=160, placeholder="اردو یا انگریزی میں خبر لکھیں...")

if st.button("🔍 اصلی AI سے چیک کریں", type="primary"):
    if not news_text.strip():
        st.warning("براہ کرم خبر لکھیں")
    else:
        with st.spinner("AI سوچ رہا ہے..."):
            # AI خود فیصلہ کرے گا کہ یہ کس کیٹیگری میں ہے
            result = ai_classifier(
                news_text,
                candidate_labels=["real authentic news", "fake sensational clickbait news"],
                hypothesis_template="This news is {}."
            )

        # result میں سب سے اوپر والا لیبل ہی AI کا فیصلہ ہے
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        fake_score = result['scores'][1] if "fake" in result['labels'][1] else result['scores'][0]
        if "fake" in top_label:
            fake_score = top_score
            real_score = result['scores'][1]
        else:
            fake_score = result['scores'][1]
            real_score = top_score

        st.markdown("---")
        if "fake" in top_label:
            st.error(f"❌ نتیجہ: یہ خبر جعلی ہے (FAKE NEWS)")
            st.metric("AI Confidence (Fake)", f"{fake_score*100:.2f}%")
            st.markdown(f"**Real ہونے کا امکان:** {real_score*100:.2f}%")
        else:
            st.success(f"✅ نتیجہ: یہ خبر اصلی ہے (REAL NEWS)")
            st.metric("AI Confidence (Real)", f"{real_score*100:.2f}%")
            st.markdown(f"**Fake ہونے کا امکان:** {fake_score*100:.2f}%")

        with st.expander("📘 AI نے کیسے فیصلہ کیا؟ (Thesis Chapter 3 & 4)"):
            st.write(f"""
            **Preprocessing:** {re.sub(r'\\s+', ' ', news_text)[:100]}...
            **Model:** XLM-RoBERTa-Large-XNLI (mBERT جیسا multilingual)
            **Reasoning:** AI نے خبر کے مطلب کو سمجھا، نہ کہ صرف لفظ دیکھے
            **Scores:** {dict(zip(result['labels'], [f'{s*100:.1f}%' for s in result['scores']]))}
            """)

st.caption("Developed by Atta Ullah | Real AI Detection | UoB 2026")
