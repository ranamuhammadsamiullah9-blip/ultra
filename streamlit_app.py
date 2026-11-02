import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Streamlit Machine Learning & Data App", layout="wide")

st.title("🍦 Breyers Products & Machine Learning Dashboard")
st.markdown("Explore your dataset, visualize target class distributions, view model metrics, and make interactive predictions.")

# Sidebar navigation
page = st.sidebar.selectbox("Choose a Page", ["Dataset Overview", "Visualizations", "Model & Predictions"])

# Load dataset (trying processed_dataset.csv or products.csv)
@st.cache_data
def load_data():
    if os.path.exists('processed_dataset.csv'):
        return pd.read_csv('processed_dataset.csv')
    elif os.path.exists('products.csv'):
        return pd.read_csv('products.csv')
    else:
        # Create a sample DataFrame if none found
        return pd.DataFrame({
            'key': ['key1', 'key2'],
            'name': ['Vanilla', 'Chocolate'],
            'description': ['Classic vanilla', 'Rich chocolate'],
            'rating': [4.5, 4.7],
            'rating_count': [100, 150],
            'ingredients': ['Milk, Sugar', 'Milk, Cocoa']
        })

df = load_data()

if page == "Dataset Overview":
    st.header("Dataset Overview")
    st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Info & Missing Values")
    col1, col2 = st.columns(2)
    with col1:
        st.text("Missing Values per Column:")
        st.write(df.isnull().sum())
    with col2:
        st.text("Data Types:")
        st.write(df.dtypes)

elif page == "Visualizations":
    st.header("Data Visualizations")
    
    if len(df.columns) > 0:
        target_col = df.columns[-1]
        st.subheader(f"Target Column Distribution ({target_col})")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        if df[target_col].nunique() <= 15 or df[target_col].dtype == 'object':
            counts = df[target_col].value_counts().head(10)
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            sns.histplot(df[target_col], kde=True, ax=ax, color='coral')
            st.pyplot(fig)
            
        st.subheader("Correlation Heatmap (Numeric Columns)")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.info("Not enough numeric columns for correlation matrix.")

elif page == "Model & Predictions":
    st.header("Model & Interactive Predictions")
    
    st.write("Here you can test predictions using the trained model pipeline.")
    
    model_path = 'trained_model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("Trained model loaded successfully!")
        
        st.subheader("Make a Prediction on Test Data")
        if st.button("Run Sample Prediction"):
            X = df.drop(columns=[df.columns[-1]]) if len(df.columns) > 1 else df
            sample = X.select_dtypes(include=[np.number]).head(1)
            if not sample.empty:
                pred = model.predict(sample)
                st.write(f"Sample Features:")
                st.dataframe(sample)
                st.write(f"Predicted Output: `{pred[0]}`")
            else:
                st.warning("No numeric feature columns available for prediction sample.")
    else:
        st.warning("No saved model file found (`trained_model.joblib`). Please run your training script first.")
