import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

# モデルの読み込み
import torch

@st.cache_resource
def load_model():
    model = torch.load("best.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("YOLOv5 分類モデル Webアプリ")
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 画像を前処理する
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)), # 学習時に使用した入力サイズにリサイズする
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 標準的なImageNetの正規化。異なる正規化を使用している場合は調整すること
    ])
    image_tensor = preprocess(image).unsqueeze(0) # バッチ次元を追加する
    # 推論
    with st.spinner("分類中..."):
        # モデルを評価モードにすることを確認する
        model.eval()
        with torch.no_grad():
            results = model(image_tensor)

        # 分類のために出力を処理する
        # モデルがクラスごとのロジット（生の出力値）または確率を出力することを前提とする
        # 最も確率の高いクラスを取得する
        probabilities = torch.nn.functional.softmax(results, dim=1)
        _, predicted_class_index = torch.max(probabilities, 1)

        # クラス名を取得 - データセットから読み込む必要があるかもしれません
        # 今のところ、予測されたクラスのインデックスだけを表示します
        st.success("分類完了！")

        # 結果表示
        st.write("### 分類結果")
        st.write(f"Predicted class index: {predicted_class_index.item()}")

        # st.write(f"Predicted class: {predicted_class_name}")
        # TODO: クラス名をインデックスの代わりに読み込んで表示する
        # 例:
        # class_names = [...] # ここでクラス名を読み込む
        # predicted_class_name = class_names[predicted_class_index.item()]
        # st.write(f"予測されたクラス: {predicted_class_name}")
