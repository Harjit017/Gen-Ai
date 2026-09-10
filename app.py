import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Telecom Complaint Semantic Search", page_icon="📡", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    data = [
        ("C001", "Aman", "Network", "My mobile network is very weak and calls keep dropping.", "Network reset and SIM reprovisioning recommended."),
        ("C002", "Priya", "Internet", "My internet connection is extremely slow during the evening.", "Checked congestion; bandwidth profile was upgraded."),
        ("C003", "Rahul", "Billing", "I was charged twice for my monthly telecom bill.", "Duplicate charge reversed and refund initiated."),
        ("C004", "Simran", "Internet", "Broadband speed is much lower than the plan I purchased.", "Router diagnostics completed and speed profile corrected."),
        ("C005", "Karan", "SIM", "My SIM card stopped working after I changed phones.", "SIM reprovisioned and network registration restored."),
        ("C006", "Neha", "Network", "Calls disconnect frequently and the signal is poor inside my house.", "Coverage issue identified; network optimization requested."),
        ("C007", "Arjun", "Internet", "WiFi keeps buffering while watching videos.", "Router restarted and channel configuration optimized."),
        ("C008", "Mehak", "Billing", "There is an unexpected roaming fee on my bill.", "Roaming charge reviewed and incorrect fee credited."),
        ("C009", "Rohit", "Data", "Mobile data is not working even though I have an active plan.", "APN settings refreshed and data service restored."),
        ("C010", "Isha", "Internet", "My broadband connection goes down several times a day.", "Line diagnostics performed; faulty connector replaced."),
        ("C011", "Vikram", "Network", "4G signal is unstable and changes to 3G repeatedly.", "Nearby cell configuration checked and device network reset."),
        ("C012", "Anjali", "Billing", "My bill amount is higher than expected this month.", "Bill itemization reviewed and disputed add-on removed."),
    ]
    return pd.DataFrame(data, columns=["complaint_id", "customer", "issue", "complaint_text", "resolution"])

@st.cache_resource
def build_embeddings(texts):
    model = load_model()
    return model.encode(texts, normalize_embeddings=True)

st.title("📡 Telecom Complaint Semantic Search")
st.caption("End-to-end vector pipeline: relational complaint data → text representation → embeddings → similarity search")

with st.sidebar:
    st.header("Pipeline")
    st.success("✓ Complaint data loaded")
    st.success("✓ Embeddings generated")
    st.success("✓ Vector similarity ready")
    st.info("Original project uses PostgreSQL + pgvector. This hosted demo keeps the vector index in memory so it can run without a managed database.")

_df = load_data()
embeddings = build_embeddings((_df["customer"] + " has " + _df["issue"] + " issue: " + _df["complaint_text"] + ". Resolution: " + _df["resolution"]).tolist())

query = st.text_input("Search telecom complaints", placeholder="e.g. slow internet problem")
k = st.slider("Number of results", 1, 10, 5)

if query:
    q = load_model().encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q
    result = _df.copy()
    result["similarity"] = scores
    result = result.sort_values("similarity", ascending=False).head(k)

    st.subheader("Top matching complaints")
    for _, row in result.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{row['complaint_id']} · {row['issue']}**")
                st.write(row["complaint_text"])
                st.markdown(f"**Resolution:** {row['resolution']}")
            with c2:
                st.metric("Similarity", f"{row['similarity']:.3f}")
else:
    st.subheader("Sample complaint data")
    st.dataframe(_df, use_container_width=True, hide_index=True)

st.divider()
st.markdown("### Architecture")
st.code("Complaint Tables → Joined Complaint Text → all-MiniLM-L6-v2 → 384-D Embeddings → Cosine Similarity → Top-K Results")
