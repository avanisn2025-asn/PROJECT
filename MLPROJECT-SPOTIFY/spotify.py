import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

st.title("🎵 Spotify Track Popularity Prediction App")

df = pd.read_csv("spotify_data clean.csv")
st.subheader("Dataset Preview")
st.dataframe(df.head())

    # Target column
target = "track_popularity"

    # Columns to drop
drop_cols = [
        "track_id",
        "track_name",
        "artist_name",
        "artist_genres",
        "album_id",
        "album_name",
        "album_release_date"
    ]

    # Remove unnecessary columns
df = df.drop(columns=drop_cols)

    # Separate features & target
X = df.drop(columns=[target])
y = df[target]

    # Identify categorical & numeric columns
categorical_cols = ["album_type"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Preprocessing
preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    # Model
model = RandomForestRegressor()

    # Pipeline
pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Split data
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
pipeline.fit(X_train, y_train)

    # Predict
y_pred = pipeline.predict(X_test)

    # Metrics
st.subheader("📊 Model Performance")
st.write("*MAE:*", round(mean_absolute_error(y_test, y_pred), 3))
st.write("*R² Score:*", round(r2_score(y_test, y_pred), 3))

st.success("Model trained successfully!")

 
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
st.pyplot(plt.gcf())




    # User input section
st.subheader("🎶 Predict Popularity for a New Track")

user_data = {}

    # Inputs for numeric features
for col in numeric_cols:
        user_data[col] = st.number_input(f"Enter {col}", value=0.0)

    # Input for album_type (categorical)
user_data["album_type"] = st.selectbox(
        "Select album_type", df["album_type"].unique()
    )

    # Predict button
if st.button("Predict Popularity"):
        input_df = pd.DataFrame([user_data])
        prediction = pipeline.predict(input_df)[0]
        st.success(f"🎯 Predicted Popularity: {round(prediction,2)}/100")
