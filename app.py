# =====================================================
# 1. IMPORT LIBRARY
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier


# =====================================================
# 2. KONFIGURASI HALAMAN & SIDEBAR
# =====================================================
st.set_page_config(page_title="PT Wira Lodya")

st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Navigasi",
    ["Beranda", "Hasil Processing Data", "Rekomendasi Pompa", "Analisis Model"]
)

# st.title("Sistem Rekomendasi Pompa Industri")
# st.subheader("PT Wira Lodya Utama")


# =====================================================
# 3. LOAD DATASET
# =====================================================
data_path = "data/Dataset_PT_WLU.xlsx"

df_products = pd.read_excel(data_path, sheet_name="Products")
df_customers = pd.read_excel(data_path, sheet_name="Customers")
df_transactions = pd.read_excel(data_path, sheet_name="Transactions")
df_transaction_details = pd.read_excel(data_path, sheet_name="Transaction_Details")

# st.success("Dataset berhasil dimuat ✅")


# =====================================================
# 4. PREPROCESSING DATA
# =====================================================
df_products = df_products.drop_duplicates(subset="product_id")
df_customers = df_customers.drop_duplicates(subset="customer_id")
df_transactions = df_transactions.drop_duplicates(subset="transaction_id")

df_transactions = df_transactions[df_transactions["status"] == "Completed"]

df_products = df_products.dropna(subset=[
    "capacity_m3h",
    "head_m",
    "power_kw",
    "price_idr"
])

# =====================================================
# 6A. COLLABORATIVE FILTERING (ITEM-BASED)
# =====================================================

# Buat user-item interaction matrix
user_item_matrix = df_transaction_details.pivot_table(
    index="transaction_id",
    columns="product_id",
    values="quantity",
    aggfunc="sum",
    fill_value=0
)

# Hitung cosine similarity antar item
item_similarity = cosine_similarity(user_item_matrix.T)

# Simpan dalam DataFrame
item_similarity_matrix = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

# =====================================================
# 5. NORMALISASI FITUR PRODUK
# =====================================================
features = ["capacity_m3h", "head_m", "power_kw", "price_idr"]

scaler = MinMaxScaler()
numeric_scaled = scaler.fit_transform(df_products[features])

df_numeric_scaled = pd.DataFrame(
    numeric_scaled,
    columns=features,
    index=df_products["product_id"]
)


# =====================================================
# 6. RANDOM FOREST FEATURE WEIGHTING
# =====================================================
# Hitung frekuensi pembelian produk
product_frequency = (
    df_transaction_details
    .groupby("product_id")
    .size()
    .reset_index(name="purchase_count")
)

# Gabungkan ke data produk
df_rf = df_products.merge(
    product_frequency,
    on="product_id",
    how="left"
)

df_rf["purchase_count"] = df_rf["purchase_count"].fillna(0)

# Label: produk laris vs tidak
threshold = df_rf["purchase_count"].median()
df_rf["label"] = (df_rf["purchase_count"] > threshold).astype(int)

X = df_rf[features]
y = df_rf["label"]

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X, y)

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_importance["importance_norm"] = (
    feature_importance["importance"] /
    feature_importance["importance"].sum()
)

feature_weights = feature_importance.set_index("feature")["importance_norm"]


# =====================================================
# 7. WEIGHTED FEATURE MATRIX
# =====================================================
df_weighted_features = df_numeric_scaled.copy()

feature_weights = feature_weights.fillna(0)

for col in df_weighted_features.columns:
    df_weighted_features[col] = (
        df_weighted_features[col] * feature_weights[col]
    )


# =====================================================
# 8. FUNGSI WEIGHTED CONTENT-BASED FILTERING
# =====================================================
def recommend_weighted_products(product_id, weighted_matrix, products_df, top_n=3):
    weighted_matrix = weighted_matrix.fillna(0)

    idx = weighted_matrix.index.get_loc(product_id)
    similarity_scores = cosine_similarity(weighted_matrix)

    sim_scores = list(enumerate(similarity_scores[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n + 1]
    product_indices = [i[0] for i in sim_scores]

    recommended_ids = weighted_matrix.index[product_indices]
    return products_df[products_df["product_id"].isin(recommended_ids)]

def hybrid_recommendation(
    product_id,
    weighted_matrix,
    cf_similarity_matrix,
    products_df,
    alpha=0.6,
    top_n=3
):
    """
    Hybrid Filtering:
    alpha * Content-Based + (1 - alpha) * Collaborative Filtering
    """

    # 🔐 Validasi product_id
    if product_id not in weighted_matrix.index:
        return pd.DataFrame()

    # ======================
    # 1️⃣ CONTENT-BASED SCORE
    # ======================
    cbf_sim = cosine_similarity(weighted_matrix)
    idx = weighted_matrix.index.get_loc(product_id)

    cbf_scores = pd.Series(
        cbf_sim[idx],
        index=weighted_matrix.index
    )

    # ======================
    # 2️⃣ COLLABORATIVE SCORE
    # ======================
    if product_id in cf_similarity_matrix.index:
        cf_scores = cf_similarity_matrix.loc[product_id]
    else:
        cf_scores = pd.Series(
            0,
            index=weighted_matrix.index
        )

    # ======================
    # 3️⃣ NORMALISASI
    # ======================
    cbf_scores = cbf_scores / cbf_scores.max() if cbf_scores.max() != 0 else cbf_scores
    cf_scores = cf_scores / cf_scores.max() if cf_scores.max() != 0 else cf_scores

    # ======================
    # 4️⃣ HYBRID SCORE
    # ======================
    hybrid_scores = alpha * cbf_scores + (1 - alpha) * cf_scores

    # 🔐 FALLBACK kalau CF kosong
    if hybrid_scores.sum() == 0:
        hybrid_scores = cbf_scores.copy()

    # ======================
    # 5️⃣ TOP-N RESULT
    # ======================
    hybrid_scores = hybrid_scores.drop(product_id, errors="ignore")
    top_products = hybrid_scores.sort_values(ascending=False).head(top_n)

    result = products_df[
        products_df["product_id"].isin(top_products.index)
    ].copy()

    result["hybrid_score"] = result["product_id"].map(top_products)

    return result.sort_values("hybrid_score", ascending=False)



# =====================================================
# 9. TAMPILAN MENU
# =====================================================
if menu == "Beranda":
    st.header("PT Wira Lodya Utama")
    st.success("Dataset berhasil dimuat ✅")
    st.write("""
    Aplikasi ini merupakan sistem rekomendasi pompa industri
    berbasis **Content-Based Filtering** yang dikombinasikan
    dengan **Random Forest Feature Weighting**.

    Sistem membantu PT Wira Lodya Utama dalam merekomendasikan
    produk pompa sesuai kebutuhan pelanggan.
    """)
    st.dataframe(df_products.head(500))


elif menu == "Hasil Processing Data":
  st.header("Hasil Processing Data")
  st.success("Preprocessing data selesai")

  st.write("Jumlah produk setelah preprocessing:", df_products.shape[0])
  st.write("Jumlah transaksi valid:", df_transactions.shape[0])

  st.subheader("Sample Data Produk (Clean)")
  st.dataframe(df_products.head())

elif menu == "Analisis Model":
    st.header("Analisis Model Random Forest")
    st.dataframe(feature_importance)


elif menu == "Rekomendasi Pompa":
    st.header("Rekomendasi Pompa Industri")

    industry = st.selectbox(
        "Industri Pelanggan",
        df_customers["industry"].dropna().unique()
    )

    capacity = st.number_input("Capacity (m3/jam)", min_value=0.0)
    head = st.number_input("Head Pompa (meter)", min_value=0.0)

    material = st.selectbox(
        "Material Pompa",
        df_products["material"].dropna().unique()
    )

    if st.button("Rekomendasikan Pompa"):
        # 🔐 SEMUA LOGIKA DI DALAM TOMBOL
        tolerance = 0.01

        filtered_df = df_products[
            (df_products["material"] == material) &
            (np.isclose(df_products["capacity_m3h"], capacity, atol=tolerance)) &
            (np.isclose(df_products["head_m"], head, atol=tolerance))
        ]

        # Tentukan produk acuan
        if filtered_df.empty:
            df_products_tmp = df_products.copy()
            df_products_tmp["distance"] = np.sqrt(
                (df_products_tmp["capacity_m3h"] - capacity) ** 2 +
                (df_products_tmp["head_m"] - head) ** 2
            )
            selected_product = df_products_tmp.sort_values("distance").iloc[0]

            if "fallback_used" not in st.session_state:
                st.session_state["fallback_used"] = False

            if filtered_df.empty and not st.session_state["fallback_used"]:
    #             st.info(
    #                 "Spesifikasi tidak ditemukan secara exact, "
    #                 "sistem menggunakan pendekatan terdekat."
    # )
                st.session_state["fallback_used"] = True
        else:
            selected_product = filtered_df.iloc[0]

        selected_product_id = selected_product["product_id"]

        # Hybrid filtering
        hybrid_result = hybrid_recommendation(
            product_id=selected_product_id,
            weighted_matrix=df_weighted_features,
            cf_similarity_matrix=item_similarity_matrix,
            products_df=df_products,
            alpha=0.6,
            top_n=10
        )

        st.subheader("⭐ Rekomendasi Pompa Terbaik (Hybrid Filtering)")
        st.dataframe(
            hybrid_result[
                [
                    "product_name",
                    "capacity_m3h",
                    "head_m",
                    "power_kw",
                    "price_idr",
                    "material",
                    "hybrid_score"
                ]
            ]
        )

        st.caption(
            "Rekomendasi dihasilkan menggunakan pendekatan Hybrid Filtering "
            "yang menggabungkan Content-Based Filtering dan Collaborative Filtering."
        )



# =====================================================
# 10. USER-ITEM INTERACTION MATRIX (CF)
# =====================================================

# Hitung frekuensi pembelian produk oleh customer
interaction_df = (
    df_transaction_details
    .groupby(["product_id", "customer_id"])
    .size()
    .reset_index(name="purchase_freq")
)

# Pivot menjadi matrix (item-based CF)
user_item_matrix = interaction_df.pivot_table(
    index="product_id",
    columns="customer_id",
    values="purchase_freq",
    fill_value=0
)

# =====================================================
# 11. ITEM-BASED COSINE SIMILARITY
# =====================================================

item_similarity_matrix = pd.DataFrame(
    cosine_similarity(user_item_matrix),
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# =====================================================
# 12. FUNGSI COLLABORATIVE FILTERING
# =====================================================

def recommend_cf_products(product_id, similarity_matrix, products_df, top_n=3):
    if product_id not in similarity_matrix.index:
        return pd.DataFrame()

    # Ambil similarity score produk
    sim_scores = similarity_matrix.loc[product_id]

    # Urutkan (kecuali dirinya sendiri)
    sim_scores = sim_scores.sort_values(ascending=False)[1:top_n + 1]

    recommended_ids = sim_scores.index

    return products_df[products_df["product_id"].isin(recommended_ids)]

# =====================================================
# 13. SCORE-BASED WEIGHTED CBF
# =====================================================

def get_cbf_scores(product_id, weighted_matrix):
    if product_id not in weighted_matrix.index:
        return pd.Series()

    similarity_matrix = cosine_similarity(weighted_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=weighted_matrix.index,
        columns=weighted_matrix.index
    )

    scores = similarity_df.loc[product_id]
    scores = scores.drop(product_id)  # hapus diri sendiri
    return scores

# =====================================================
# 14. SCORE-BASED CF
# =====================================================

def get_cf_scores(product_id, similarity_matrix):
    if product_id not in similarity_matrix.index:
        return pd.Series()

    scores = similarity_matrix.loc[product_id]
    scores = scores.drop(product_id)
    return scores
def normalize_scores(scores):
    if scores.empty:
        return scores

    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

# =====================================================
# 15. HYBRID FILTERING FUNCTION
# =====================================================

def hybrid_recommendation(
    product_id,
    weighted_matrix,
    cf_similarity_matrix,
    products_df,
    alpha=0.6,
    top_n=3
):
    # Ambil score CBF & CF
    cbf_scores = get_cbf_scores(product_id, weighted_matrix)
    cf_scores = get_cf_scores(product_id, cf_similarity_matrix)

    # Normalisasi
    cbf_scores = normalize_scores(cbf_scores)
    cf_scores = normalize_scores(cf_scores)

    # Gabungkan index
    all_products = cbf_scores.index.union(cf_scores.index)

    hybrid_scores = {}

    for pid in all_products:
        cbf_score = cbf_scores.get(pid, 0)
        cf_score = cf_scores.get(pid, 0)

        hybrid_scores[pid] = alpha * cbf_score + (1 - alpha) * cf_score

    hybrid_scores = pd.Series(hybrid_scores).sort_values(ascending=False)

    top_products = hybrid_scores.head(top_n)

    result = products_df[
        products_df["product_id"].isin(top_products.index)
    ].copy()

    result["hybrid_score"] = result["product_id"].map(top_products)

    return result.sort_values("hybrid_score", ascending=False)
