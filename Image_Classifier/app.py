import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from streamlit_drawable_canvas import st_canvas
import cv2

# ------------------------------
# Load MNIST dataset
# ------------------------------
@st.cache_data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # Normalize
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Reshape
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # One-hot encode labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


# ------------------------------
# Build CNN model
# ------------------------------
def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ------------------------------
# Helper: Plot probabilities (small)
# ------------------------------
def plot_probabilities(pred):
    fig, ax = plt.subplots(figsize=(4, 2))  # compact
    ax.bar(range(10), pred[0])
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit", fontsize=8)
    ax.set_ylabel("Prob", fontsize=8)
    ax.set_ylim([0, 1])
    ax.set_title("Probabilities", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    return fig


# ------------------------------
# Helper: preprocess a single digit crop -> 28x28 centered (MNIST-like)
# ------------------------------
def preprocess_crop(crop):
    # crop is grayscale uint8
    # Normalize foreground as white on black like MNIST (digits bright)
    # If background is white (from canvas), invert to make digit white on black
    # (Our canvas uses black background + white stroke, so this is already correct.)
    # Resize while preserving aspect ratio, pad to square, center by moments.
    # 1) binarize
    _, th = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2) get tight bbox again to remove excess padding
    ys, xs = np.where(th > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    digit = th[y1:y2+1, x1:x2+1]

    # 3) make square by padding
    h, w = digit.shape
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    squared = cv2.copyMakeBorder(digit, pad_top, pad_bottom, pad_left, pad_right,
                                 cv2.BORDER_CONSTANT, value=0)

    # 4) resize to 20x20 then pad to 28x28 (MNIST style)
    resized = cv2.resize(squared, (20, 20), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(resized, 4, 4, 4, 4,
                                cv2.BORDER_CONSTANT, value=0)

    # 5) center via center of mass shift
    # compute moments
    M = cv2.moments(padded)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        shiftx = int(np.round(14 - cx))
        shifty = int(np.round(14 - cy))
        Mshift = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        centered = cv2.warpAffine(padded, Mshift, (28, 28), flags=cv2.INTER_NEAREST, borderValue=0)
    else:
        centered = padded

    img = centered.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img


# ------------------------------
# Helper: segment multi-digit drawing into ordered crops
# ------------------------------
def segment_digits_from_canvas(rgba_img, min_area=50):
    """
    rgba_img: np.uint8 (h,w,4) from st_canvas
    Returns list of (x, crop_gray) sorted left->right
    """
    # Convert to grayscale (digit is white, background is black if we set canvas that way)
    gray = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2GRAY)

    # Slight blur helps contours
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binary (OTSU)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: close small holes
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours (external)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h, w = th.shape
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < min_area:
            continue
        # Slightly expand box to include margins
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + cw + pad)
        y1 = min(h, y + ch + pad)
        crop = gray[y0:y1, x0:x1]
        boxes.append((x0, crop))

    # Sort left->right
    boxes.sort(key=lambda t: t[0])
    return boxes


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="MNIST Classifier", layout="wide")
st.title("🖼️ MNIST Digit Classifier with CNN")
st.caption("Left: draw a digit or multiple digits. Right: compact training graph.")

# Load data
train_images, train_labels, test_images, test_labels = load_data()

# Sidebar controls
st.sidebar.header("⚙️ Model Settings")
epochs = st.sidebar.slider("Epochs", 1, 10, 5)
batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128], index=1)
train_button = st.sidebar.button("🚀 Train / Retrain Model")

# Session state for model
if "model" not in st.session_state:
    st.session_state.model = None
if "history" not in st.session_state:
    st.session_state.history = None

# Train the model
if train_button:
    st.session_state.model = build_model()
    with st.spinner("Training the model..."):
        history = st.session_state.model.fit(
            train_images, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_images, test_labels),
            verbose=1
        )
        st.session_state.history = history.history
    st.success("✅ Training complete!")

# ------------------------------
# Layout: Left (canvas & prediction) | Right (training graph)
# ------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("✏️ Draw a Digit (or Multiple Digits)")
    st.caption("Use a thick stroke and leave small gaps between digits for best results.")

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=12,
        stroke_color="white",
        background_color="black",  # black background, white strokes (MNIST-like)
        width=280,
        height=200,
        drawing_mode="freedraw",
        key="canvas_multi",
    )

    cols_action = st.columns([1, 1, 2])
    with cols_action[0]:
        predict_btn = st.button("Predict")
    with cols_action[1]:
        clear_btn = st.button("Clear")

    if clear_btn:
        st.experimental_rerun()

    if predict_btn:
        if st.session_state.model is None:
            st.warning("Please train the model first from the sidebar.")
        elif canvas_result.image_data is None:
            st.info("Draw something first.")
        else:
            rgba = canvas_result.image_data.astype(np.uint8)
            boxes = segment_digits_from_canvas(rgba, min_area=80)

            if not boxes:
                st.info("No digits detected. Try drawing thicker/clearer.")
            else:
                preds_text = []
                pred_probs = []
                digit_images = []

                for x, crop in boxes:
                    proc = preprocess_crop(crop)
                    if proc is None:
                        continue
                    pred = st.session_state.model.predict(proc, verbose=0)
                    digit = int(np.argmax(pred))
                    preds_text.append(str(digit))
                    pred_probs.append(pred)
                    digit_images.append(proc.reshape(28, 28))

                if len(preds_text) == 0:
                    st.info("Could not extract clear digits. Try again.")
                else:
                    st.success(f"🧠 Predicted Sequence: **{''.join(preds_text)}**")

                    # Show each digit + probability compactly
                    for i, (img28, p) in enumerate(zip(digit_images, pred_probs), start=1):
                        c1, c2 = st.columns([1, 1.4])
                        with c1:
                            st.image(img28, caption=f"Digit {i}: {np.argmax(p)}", width=100, use_container_width=False)
                        with c2:
                            st.pyplot(plot_probabilities(p))

with right:
    if st.session_state.history:
        st.subheader("📈 Training Progress")
        fig, ax = plt.subplots(figsize=(3.4, 2.0))  # smaller chart on the right
        ax.plot(st.session_state.history['accuracy'], label='Train Acc')
        ax.plot(st.session_state.history['val_accuracy'], label='Val Acc')
        ax.legend(fontsize=7)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    # Optional: quick MNIST test image prediction (compact)
    if st.session_state.model is not None:
        st.markdown("---")
        st.caption("Quick test (MNIST): pick index and see prediction")
        idx = st.slider("Test index", 0, len(test_images) - 1, 0, key="test_idx_side")
        pred_m = st.session_state.model.predict(np.expand_dims(test_images[idx], axis=0), verbose=0)
        st.image(test_images[idx].reshape(28, 28), caption=f"Pred: {np.argmax(pred_m)}", width=120, use_container_width=False)
