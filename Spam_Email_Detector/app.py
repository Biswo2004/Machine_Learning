import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ------------------------------
# Load and preprocess dataset
# ------------------------------
@st.cache_data
def load_data():
    # SMSSpamCollection format: label \t message
    data = pd.read_csv("SMSSpamCollection.csv", sep="\t", header=None, names=["label", "text"])
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Convert ham/spam to 0/1
    return data

data = load_data()

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# ------------------------------
# Train model with TF-IDF
# ------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions for evaluation
y_pred = model.predict(X_test_vec)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="📧 Spam Detector", layout="wide")

st.title("📧 Spam Email/SMS Detector")
st.write("A simple app that detects whether a message is **Spam** or **Ham (Not Spam)**.")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Mode:", ["Single Message Detection", "Bulk Detection", "Model Evaluation"])

# ------------------------------
# Single Message Detection
# ------------------------------
if app_mode == "Single Message Detection":
    st.subheader("🔍 Check a Single Message")
    user_input = st.text_area("Enter the email/SMS text below:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message first.")
        else:
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]
            label = "🚨 Spam" if prediction == 1 else "✅ Not Spam"
            st.markdown(f"### Prediction: {label}")

# ------------------------------
# Bulk Detection
# ------------------------------
elif app_mode == "Bulk Detection":
    st.subheader("📂 Bulk Upload for Spam Detection")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("Uploaded CSV must have a 'text' column!")
        else:
            X_bulk = vectorizer.transform(df['text'])
            df['prediction'] = model.predict(X_bulk)
            df['prediction'] = df['prediction'].map({0: "Not Spam", 1: "Spam"})
            st.write(df.head(10))
            
            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

# ------------------------------
# Model Evaluation
# ------------------------------
elif app_mode == "Model Evaluation":
    st.subheader("📊 Model Performance")

    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write("**Accuracy:**", accuracy)
    st.write("**Precision:**", precision)
    st.write("**Recall:**", recall)
    st.write("**F1 Score:**", f1)

    # Confusion Matrix
    st.write("### Confusion Matrix")
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

# ------------------------------
# Footer
# ------------------------------
st.markdown(
    """
    <hr>
    <div style="text-align: center; color: gray; font-size: 14px;">
        📧 Spam Detector App | Built with ❤️ using <b>Streamlit</b><br>
        👨‍💻 Developed by <b>Your Name</b>
    </div>
    """,
    unsafe_allow_html=True
)
