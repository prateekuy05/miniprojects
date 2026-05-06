
import os
import requests
from io import BytesIO
from urllib.parse import urlparse
from typing import Any

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess



def is_http_url(candidate: Any) -> bool:
    """Return **True** if *candidate* is a non‑empty string that begins with http/https."""
    return isinstance(candidate, str) and urlparse(candidate).scheme in {"http", "https"}


def safe_image_exists(path: Any) -> bool:
    """Return **True** if *path* is a non‑empty string pointing to an existing local file."""
    return isinstance(path, str) and path.strip() != "" and os.path.exists(path)




def extract_image_features(img_source: Any, model: ResNet50) -> np.ndarray:
    """
    Return a 2 048‑D feature vector from an image.

    • `img_source` can be an HTTP/HTTPS URL **or** a local file path.
    • On any error (including NaN / non‑string), an all‑zeros vector of the correct length is returned.
    """
    try:
        if not isinstance(img_source, str) or img_source.strip() == "":
            raise ValueError("img_source must be a non‑empty string")

        if is_http_url(img_source):
            resp = requests.get(img_source, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            img = keras_image.load_img(img_source, target_size=(224, 224))

        img = img.resize((224, 224))
        x = keras_image.img_to_array(img)[None, ...]
        x = resnet_preprocess(x)

        feats = model.predict(x, verbose=0)  # (1, 2048)
        return feats.flatten()

    except Exception as e:
        print(f"[extract_image_features] Failed on {img_source}: {e}")
        return np.zeros((model.output_shape[1],), dtype="float32")



@st.cache_data
def load_data() -> pd.DataFrame:
    """Load product catalogue and pre‑compute (or load) image features."""
    df = pd.read_csv("products.csv")

    if "image" not in df.columns:
        raise KeyError("products.csv must contain an 'image' column.")
    df["image"] = df["image"].fillna("")

    if "product_name" not in df.columns:
        raise KeyError("products.csv must contain a 'product_name' column.")
    df["product_name"] = df["product_name"].fillna("").astype(str)

    model = get_resnet_model()
    df["image_features"] = df["image"].apply(lambda p: extract_image_features(p, model))
    return df


@st.cache_data
def load_users() -> pd.DataFrame:
    """Load user credentials and standardise column names."""
    df = pd.read_csv("users.csv")
    df.columns = df.columns.str.strip().str.lower()
    if not {"username", "password"}.issubset(df.columns):
        raise KeyError("users.csv must contain 'username' and 'password' columns.")
    return df


@st.cache_resource 
def get_resnet_model() -> ResNet50:
    """Global ResNet‑50 feature extractor (avg‑pooled)."""
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")


def authenticate(username: str, password: str, users_df: pd.DataFrame) -> bool:
    """Plain‑text username/password check (demo‑only)."""
    match = users_df[(users_df["username"] == username) & (users_df["password"] == password)]
    return not match.empty


def build_text_similarity(df: pd.DataFrame) -> np.ndarray:
    """Cosine‑similarity matrix for product descriptions."""
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["description"].fillna(""))
    return cosine_similarity(tfidf_matrix)


def recommend(product_name: str, df: pd.DataFrame, text_sim: np.ndarray) -> pd.DataFrame:
    """Return top‑5 products similar in **both** description **and** image."""
    if product_name not in df["product_name"].values:
        return pd.DataFrame()

    idx_query = df.index[df["product_name"] == product_name][0]

    sim_scores = list(enumerate(text_sim[idx_query]))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    text_matches = sim_scores[1:11]  # exclude the query itself

   
    model = get_resnet_model()
    q_img_vec = df.at[idx_query, "image_features"]

    combined: list[tuple[int, float]] = []
    for idx_candidate, text_score in text_matches:
        cand_img_vec = df.at[idx_candidate, "image_features"]
        img_score = cosine_similarity([q_img_vec], [cand_img_vec])[0][0]
        combined.append((idx_candidate, 0.5 * text_score + 0.5 * img_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in combined[:5]]
    return df.loc[top_indices]



st.set_page_config(page_title="Product Recommendation App", layout="centered")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

if not st.session_state["authenticated"]:
    st.title("🔒 Login to Access the Product Recommendation System")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        users_df = load_users()
        if authenticate(username, password, users_df):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password.")


else:
    st.success(f"Welcome, **{st.session_state['username']}**!")

    df_products = load_data()
    text_sim_matrix = build_text_similarity(df_products)

    st.header("🛍 Product Recommendation System")

    category_choices = ["All"] + sorted(df_products["category"].dropna().unique())
    selected_cat = st.selectbox("Filter by category:", category_choices)

    filtered_df = df_products if selected_cat == "All" else df_products[df_products["category"] == selected_cat]

    product_choices = filtered_df["product_name"].tolist()
    selected_product = st.selectbox("Choose a product:", product_choices)

    if st.button("Get Recommendations"):
        results = recommend(selected_product, df_products, text_sim_matrix)

        if results.empty:
            st.warning("No similar products found.")
        else:
            st.subheader("Recommended Products:")
            for _, row in results.iterrows():
                img_src = row["image"]
                if is_http_url(img_src):
                    st.image(img_src, width=160)
                elif safe_image_exists(img_src):
                    st.image(img_src, width=160)
                else:
                    st.write("(No image available)")

                st.markdown(f"**{row['product_name']}**")
                st.caption(f"Category: {row['category']}")
                st.write(row["description"])
                st.markdown("---")

    if st.checkbox("Show full dataset"):
        st.dataframe(df_products.drop(columns=["image_features"]))
