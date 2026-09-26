import streamlit as st

st.set_page_config(page_title="حقیقت.pk", page_icon="🇵🇰", layout="centered")

st.markdown("<h1 style='text-align: center;'>حقیقت.pk</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI سے جعلی خبروں کی پہچان - سچ اور جھوٹ میں فرق جانیں</p>", unsafe_allow_html=True)
st.divider()

news_text = st.text_area("یہاں خبر کا متن لکھیں:", height=150, placeholder="مثال: حکومت نے اعلان کیا ہے کہ...")

if st.button("حقیقت چیک کریں 🔍", use_container_width=True):
    if not news_text.strip():
        st.warning("براہ کرم پہلے خبر لکھیں۔")
    else:
        fake_keywords = ["100% سچ", "فوری شیئر کریں", "یقین نہیں آئے گا", "حکومت گر گئی", "خفیہ", "لازمی دیکھیں"]
        score = 0
        for word in fake_keywords:
            if word in news_text:
                score += 20
        
        if score >= 40:
            st.error(f⚠️ نتیجہ: یہ خبر مشکوک / جعلی ہو سکتی ہے ({score}% امکان)")
            st.write("اس میں سنسنی پھیلانے والے الفاظ ہیں۔ تصدیق کے بغیر شیئر نہ کریں۔")
        elif score >= 20:
            st.warning(f"🟡 نتیجہ: تصدیق کی ضرورت ہے ({score}% مشکوک)")
            st.write("اس خبر کی کسی مستند ذریعے سے تصدیق کر لیں۔")
        else:
            st.success(f"✅ نتیجہ: یہ خبر درست لگ رہی ہے ({100-score}% درست امکان)")
            st.write("اس میں کوئی مشکوک پیٹرن نہیں ملا، پھر بھی سرکاری ذرائع سے چیک کر لیں۔")

st.divider()
st.caption("نوٹ: یہ ابتدائی ورژن ہے، ہم جلد اس میں بڑا AI ماڈل شامل کریں گے۔")