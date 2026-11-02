import streamlit as st
import pickle
import pandas as pd

# The pickled artifact is now a FULL sklearn Pipeline:
# ColumnTransformer (OneHotEncoder for categoricals, median imputer for numerics) + XGBoost.
# Raw category labels are sent directly — no manual integer mapping needed.
FEATURES = [
    "neighbourhood_group", "room_type", "latitude", "longitude",
    "minimum_nights", "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
]


def main():
    st.title("🏡 Airbnb Price Prediction")
    st.write("Enter the details below to predict the price of an Airbnb listing.")

    neighbourhood_group = st.selectbox(
        "Neighbourhood Group",
        ["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"],
    )
    latitude = st.number_input("Latitude", value=40.7128)
    longitude = st.number_input("Longitude", value=-74.0060, max_value=0.0)
    room_type = st.selectbox(
        "Room Type", ["Entire home/apt", "Private room", "Shared room"]
    )
    minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=1)
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=5)
    reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=1.0)
    calculated_host_listings_count = st.number_input("Host Listings Count", min_value=1, value=1)
    availability_365 = st.slider("Availability (Days in a Year)", 0, 365, 180)

    input_data = pd.DataFrame([{
        "neighbourhood_group": neighbourhood_group,
        "room_type": room_type,
        "latitude": latitude,
        "longitude": longitude,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
    }])[FEATURES]

    try:
        with open("best_price_model.pkl", "rb") as file:
            model = pickle.load(file)

        if st.button("Predict Price"):
            prediction = model.predict(input_data)[0]
            st.success(f"💰 Estimated Price: ${prediction:,.2f} per night")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'best_price_model.pkl' is in the same directory.")
    except (ValueError, KeyError) as e:
        st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
