import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Load model and vectorizer
model = joblib.load("review_mnb_model.pkl")
vectorizer = joblib.load("bow_vectorizer.pkl")

# Custom Title
st.markdown("<h1 style='text-align: center;'>🎬 Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze whether a review is Positive or Negative</p>", unsafe_allow_html=True)

st.divider()

# Input box
review = st.text_area("✍️ Enter your movie review below:", height=150)

# Analyze Button
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if review.strip():

        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)[0]
        probabilities = model.predict_proba(review_vec)[0]
        confidence = max(probabilities)

        st.divider()

        # Display Result
        if prediction == 1:
            st.success("🎉 Positive Review")
        else:
            st.error("😞 Negative Review")

        # Show Confidence Score
        st.metric("Confidence Score", f"{round(confidence * 100, 2)} %")

        # Show Probability Breakdown
        st.subheader("Prediction Breakdown")
        st.progress(float(probabilities[1]))  # Positive probability
        st.write(f"Positive Probability: {round(probabilities[1] * 100, 2)}%")
        st.write(f"Negative Probability: {round(probabilities[0] * 100, 2)}%")

    else:
        st.warning("⚠️ Please enter a review before analyzing.")
