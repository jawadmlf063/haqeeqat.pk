import streamlit as st

st.set_page_config(page_title="haqeeqat.pk - Sach ki Pehchan", layout="centered")

st.title("haqeeqat.pk")
st.subheader("AI se Jaali Khabron ki Pehchan")
st.divider()

news_text = st.text_area("Yahan khabar ka matan likhen:", height=200, placeholder="Yahan khabar paste karen...")

col1, col2 = st.columns(2)
with col1:
    check_btn = st.button("Haqeeqat Check Karen", use_container_width=True, type="primary")
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if check_btn:
    if not news_text.strip():
        st.warning("Barae meherbani pehle khabar likhen.")
    else:
        fake_keywords = ["100% sach", "fori share karen", "yaqeen nahi aye ga", "hukumat gir gayi", "khufia inkishaf", "tez tareen"]
        score = 0
        found = []
        for word in fake_keywords:
            if word.lower() in news_text.lower():
                score += 20
                found.append(word)
        
        st.divider()
        if score >= 60:
            st.error(f"Natija: Ye khabar mashkook / jaali ho sakti hai ({score}%)")
        elif score >= 20:
            st.warning(f"Natija: Is khabar ki mazeed tasdeeq ki zaroorat hai ({score}% mashkook)")
        else:
            st.success(f"Natija: Ye khabar durust lag rahi hai ({100-score}% durust)")

        if found:
            st.write("Mashkook alfaz mile:", ", ".join(found))

st.divider()
st.caption("Developed by jawadmlf063 | haqeeqat.pk")
