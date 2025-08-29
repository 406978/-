import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

# クラス名
class_names = ["玉石", "ガンタ石", "大谷石", "間知石", "RC"]

# モデル構造
class SimpleClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 128 * 3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, len(class_names))
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    try:
        model = SimpleClassifier()
        model.load_state_dict(torch.load("best.pt", map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        return None

model = load_model()

# Streamlit UI
st.title("YOLOv5 分類モデル Webアプリ")
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image).unsqueeze(0)

    with st.spinner("分類中..."):
        try:
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class_index = torch.argmax(probabilities, dim=1).item()
                predicted_class = class_names[predicted_class_index]
                confidence = probabilities[0][predicted_class_index].item()

            st.success("分類完了！")
            st.write(f"**予測されたクラス:** {predicted_class}")
            st.write(f"**信頼度:** {confidence:.2f}")
        except Exception as e:
            st.error(f"分類中にエラーが発生しました: {e}")

# 棒グラフの表示
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(class_names, probabilities.squeeze().numpy(), color='skyblue')
ax.set_ylabel("確率")
ax.set_title("分類結果の確率分布")
ax.set_ylim([0, 1])
st.pyplot(fig)
            
        
