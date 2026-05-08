
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from google.colab import files
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP MODEL
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3, activation='softmax')
])

labels = ['Dry', 'Normal', 'Oily']

# 2. INTENSITY & RECOMMENDATION ENGINE
def analyze_skin_depth(skin_type, confidence):
    # Determine Level based on AI Confidence
    if confidence > 0.85:
        level = "High"
    elif confidence > 0.65:
        level = "Moderate"
    else:
        level = "Mild"

    # Mapping logic for all types
    data = {
        'Oily': {
            'High': ("Severely Oily", "Deep-cleansing clay masks & 2% BHA Salicylic acid."),
            'Moderate': ("Moderately Oily", "Oil-control foaming cleanser & Niacinamide serum."),
            'Mild': ("Slightly Oily", "Lightweight gel-based moisturizer & double cleansing.")
        },
        'Dry': {
            'High': ("Extremely Dehydrated", "Rich ceramide balms, facial oils, and no-foam cleansers."),
            'Moderate': ("Noticeably Dry", "Hyaluronic acid serum and thick cream-based moisturizers."),
            'Mild': ("Slightly Dry", "Hydrating toner and a standard creamy moisturizer.")
        },
        'Normal': {
            'High': ("Perfectly Balanced", "Maintain with SPF 50 and a gentle antioxidant serum."),
            'Moderate': ("Healthy / Normal", "Standard cleanse, moisturize, and protect routine."),
            'Mild': ("Mostly Normal", "Focus on maintaining the skin barrier with gentle care.")
        }
    }

    return data[skin_type][level]

# 3. EXECUTION
def run_glow_ai_full():
    print("--- GlowAI: Full Spectrum Skin Analysis ---")
    uploaded = files.upload()

    for fn in uploaded.keys():
        img = image.load_img(fn, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # AI Prediction
        preds = model.predict(img_array)[0]
        max_idx = np.argmax(preds)
        conf_score = preds[max_idx]
        detected_type = labels[max_idx]

        # Get Detailed Analysis
        intensity_label, advice = analyze_skin_depth(detected_type, conf_score)

        # Visual Output
        plt.imshow(img)
        plt.title(f"GlowAI: {intensity_label}")
        plt.axis('off')
        plt.show()

        print("=" * 40)
        print(f"📊 ANALYSIS FOR: {fn.upper()}")
        print(f"🔹 PRIMARY TYPE : {detected_type}")
        print(f"🔹 INTENSITY    : {intensity_label}")
        print(f"🔹 AI CONFIDENCE : {conf_score*100:.1f}%")
        print(f"💡 GLOW-ADVICE  : {advice}")
        print("=" * 40)

run_glow_ai_full()
