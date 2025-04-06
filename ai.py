import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config(page_title="ML & AI Explorer", layout="wide")
st.title("Machine Learning and AI Explorer")

section = st.sidebar.selectbox("Select Section", ["Regression", "Clustering", "Neural Network"])

# --- Regression Section ---
if section == "Regression":
    st.header("Regression Model")
    uploaded_file = st.file_uploader("Upload CSV Dataset for Regression", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        target = st.text_input("Enter the name of the target column")
        if target and target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]

            # Optional preprocessing: drop NA
            if st.checkbox("Drop NA rows"):
                df = df.dropna()
                X = df.drop(columns=[target])
                y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("Model Performance")
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
            st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Predicted vs Actual")
            st.pyplot(fig)

            st.subheader("Custom Prediction")
            input_data = {}
            for feature in X.columns:
                input_data[feature] = st.number_input(f"{feature}", float(X[feature].min()), float(X[feature].max()))
            if st.button("Predict"):
                custom_input = np.array([list(input_data.values())])
                prediction = model.predict(custom_input)
                st.write(f"Predicted Value: {prediction[0]:.2f}")

# --- Clustering Section ---
elif section == "Clustering":
    st.header("K-Means Clustering")
    uploaded_file = st.file_uploader("Upload CSV Dataset for Clustering", type=["csv"], key="clustering")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        cluster_data = st.multiselect("Select numeric columns for clustering", df.select_dtypes(include=np.number).columns.tolist())
        if cluster_data:
            k = st.slider("Select number of clusters", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(df[cluster_data])
            df['Cluster'] = clusters

            if len(cluster_data) == 2:
                fig = px.scatter(df, x=cluster_data[0], y=cluster_data[1], color=df['Cluster'].astype(str), title="2D Cluster Plot")
                st.plotly_chart(fig)
            elif len(cluster_data) == 3:
                fig = px.scatter_3d(df, x=cluster_data[0], y=cluster_data[1], z=cluster_data[2], color=df['Cluster'].astype(str), title="3D Cluster Plot")
                st.plotly_chart(fig)

            st.download_button("Download Clustered Dataset", df.to_csv(index=False).encode(), "clustered_data.csv", "text/csv")

# --- Neural Network Section ---
elif section == "Neural Network":
    st.header("Feedforward Neural Network")
    uploaded_file = st.file_uploader("Upload Classification Dataset (CSV)", type=["csv"], key="nn")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        target = st.text_input("Enter target column name")
        if target and target in df.columns:
            X = df.drop(columns=[target])
            y = pd.get_dummies(df[target])

            if st.checkbox("Drop NA rows", key="nn_na"):
                df = df.dropna()
                X = df.drop(columns=[target])
                y = pd.get_dummies(df[target])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.subheader("Model Training Parameters")
            epochs = st.slider("Epochs", 1, 100, 10)
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, step=0.0001)

            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(y.shape[1], activation='softmax')
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=0)

            st.subheader("Training Progress")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history.history['loss'], label='Train Loss')
            ax[0].plot(history.history['val_loss'], label='Val Loss')
            ax[0].set_title('Loss')
            ax[0].legend()
            ax[1].plot(history.history['accuracy'], label='Train Acc')
            ax[1].plot(history.history['val_accuracy'], label='Val Acc')
            ax[1].set_title('Accuracy')
            ax[1].legend()
            st.pyplot(fig)

            st.subheader("Upload Custom Sample for Prediction")
            sample_file = st.file_uploader("Upload sample for prediction (CSV)", type=["csv"], key="sample")
            if sample_file:
                sample_df = pd.read_csv(sample_file)
                preds = model.predict(sample_df)
                pred_labels = y.columns[np.argmax(preds, axis=1)]
                sample_df['Prediction'] = pred_labels
                st.dataframe(sample_df)
