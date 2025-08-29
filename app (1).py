import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

# モデルの読み込み
@st.cache_resource
def load_model():
    # Correct the path to hubconf.py
    model = torch.hub.load('/content/yolov5', 'custom', path='/content/yolov5/runs/train-cls/exp/weights/best.pt', source='local', force_reload=True)
    return model

model = load_model()

# Streamlit UI
st.title("YOLOv5 分類モデル Webアプリ")
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)), # Resize to the input size used for training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization, adjust if different was used
    ])
    image_tensor = preprocess(image).unsqueeze(0) # Add batch dimension

    # 推論
    with st.spinner("分類中..."):
        # Ensure model is in evaluation mode
        model.eval()
        with torch.no_grad():
            results = model(image_tensor)

        # Process the output for classification
        # Assuming the model outputs logits or probabilities for classes
        # Get the class with the highest probability
        probabilities = torch.nn.functional.softmax(results, dim=1)
        _, predicted_class_index = torch.max(probabilities, 1)

        # Get class names - you might need to load these from your dataset
        # For now, I'll just display the predicted class index
        st.success("分類完了！")

        # 結果表示
        st.write("### 分類結果")
        st.write(f"Predicted class index: {predicted_class_index.item()}")

        # TODO: Load and display class names instead of index
        # Example:
        # class_names = [...] # Load your class names here
        # predicted_class_name = class_names[predicted_class_index.item()]
        # st.write(f"Predicted class: {predicted_class_name}")
