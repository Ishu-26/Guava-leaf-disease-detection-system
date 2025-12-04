import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# # -------------------------
# # Load TFLite model
# # -------------------------
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# IMG_SIZE = 224   # MobileNet input size

# # Correct 6 classes
# CLASS_NAMES = [
#     "anthracnose",
#     "healthy",
#     "insect_bite",
#     "multiple",
#     "scorch",
#     "yld"
# ]

# # -------------------------
# # Preprocess Image (FLOAT32)
# # -------------------------
# def preprocess_image(image):
#     image = image.convert("RGB")
#     image = image.resize((IMG_SIZE, IMG_SIZE))

#     img_array = np.array(image, dtype=np.float32)
#     img_array = img_array / 255.0  # normalize for float32 model
#     img_array = np.expand_dims(img_array, axis=0)

#     return img_array

# # -------------------------
# # Prediction Function
# # -------------------------
# def predict(image):
#     if image is None:
#         return "No image uploaded."

#     img_array = preprocess_image(image)

#     interpreter.set_tensor(input_details[0]["index"], img_array)
#     interpreter.invoke()

#     output = interpreter.get_tensor(output_details[0]["index"])[0]

#     predicted_class = CLASS_NAMES[np.argmax(output)]
#     confidence = float(np.max(output)) * 100

#     return f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"

# # -------------------------
# # Gradio UI
# # -------------------------
# app = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil", label="Upload Leaf Image"),
#     outputs=gr.Textbox(label="Prediction Result"),
#     title="Plant Leaf Disease Classifier",
#     description="Upload a guava leaf to detect Anthracnose, Insect Bites, Scorch, Healthy, Multiple, or YLD."
# )

# app.launch()















import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# Load TFLite model
# -------------------------
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 224

CLASS_NAMES = [
    "anthracnose",
    "healthy",
    "insect_bite",
    "multiple",
    "scorch",
    "yld"
]

# -------------------------
# Preprocess
# -------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -------------------------
# Plot Bar Chart
# -------------------------
def plot_probabilities(probabilities):
    plt.figure(figsize=(6, 3))
    plt.bar(CLASS_NAMES, probabilities)
    plt.xticks(rotation=30)
    plt.title("Class Probability Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.tight_layout()

    plt.savefig("prob.png")
    plt.close()
    return "prob.png"

CLASS_INFO = {
    "anthracnose": "Cause: Fungal infection. Prevention: Avoid overhead watering, use resistant varieties. Treatment: Apply fungicides.",
    "healthy": "This leaf is healthy! Keep monitoring and maintain proper irrigation and nutrition.",
    "insect_bite": "Cause: Insect attack (e.g., aphids, caterpillars). Prevention: Use neem oil or insecticidal sprays. Remove affected leaves.",
    "multiple": "Multiple diseases detected. Prevention: Isolate affected plants, improve hygiene, rotate crops. Apply fungicides/insecticides as needed.",
    "scorch": "Cause: Excess heat, sunburn, or nutrient deficiency. Prevention: Shade sensitive plants, maintain proper watering. Treatment: Provide nutrients and water.",
    "yld": "Yellow Leaf Disease: Caused by viral infection. Prevention: Remove infected plants, control insect vectors. Use certified disease-free seeds/seedlings."
}


# -------------------------
# Prediction Function
# -------------------------
def predict(image):
    if image is None:
        return "Upload an image", None, None, "Awaiting analysis..."

    img_array = preprocess_image(image)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted_class = CLASS_NAMES[np.argmax(output)]
    confidence = float(np.max(output)) * 100

    prob_plot_path = plot_probabilities(output)

    result_text = (
        f"### 🟢 Prediction: **{predicted_class.upper()}**\n"
        f"Confidence: **{confidence:.2f}%**"
    )

    # Bold info text for the separate block
    info_text = f"### 📌 Info / Remedies for {predicted_class.upper()}:\n\n**{CLASS_INFO.get(predicted_class, 'No information available.')}**"

    return result_text, prob_plot_path, image, info_text


# -------------------------
# UI (Load external CSS)
# -------------------------
css_data = open("style.css", "r").read()

with gr.Blocks() as app:

    # inject CSS
    gr.HTML(f"<style>{css_data}</style>")

    # Title and subtitle
    gr.HTML("<div class='title'>🌿 Guava Leaf Disease Detector</div>")
    gr.HTML("<div class='sub'>AI-powered detection of 6 plant diseases with MobileNetV2</div>")

    # Main row
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type='pil', label="Upload Leaf Image")
            btn = gr.Button("🔍 Analyze Leaf", variant="primary")

        with gr.Column(scale=1):
            result = gr.Markdown("Prediction will appear here...")
            graph = gr.Image(label="Probability Distribution")
            preview = gr.Image(label="Uploaded Image Preview")
            info_block = gr.Markdown("Info / Remedies will appear here...")  # NEW BLOCK

    # Attach button click inside the Blocks context
    btn.click(fn=predict, inputs=inp, outputs=[result, graph, preview, info_block])

# Launch app
app.launch()

