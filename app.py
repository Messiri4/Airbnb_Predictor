import streamlit as st
import pickle  
import numpy as np

def main():
    st.title("🏡 Airbnb Price Prediction")
    st.write("Enter the details below to predict the price of an Airbnb listing.")

    # User inputs
    neighbourhood_group = st.selectbox("Neighbourhood Group", ["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"])
    latitude = st.number_input("Latitude", value=40.7128)  # Default to NYC coordinates
    longitude = st.number_input("Longitude", value=-74.0060)
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
    minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=1)
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=5)
    reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=1.0)
    calculated_host_listings_count = st.number_input("Host Listings Count", min_value=1, value=1)
    availability_365 = st.slider("Availability (Days in a Year)", 0, 365, 180)

    # Convert categorical inputs
    room_type_map = {"Entire home/apt": 0, "Private room": 1, "Shared room": 2}
    neighbourhood_group_map = {"Brooklyn": 0, "Manhattan": 1, "Queens": 2, "Bronx": 3, "Staten Island": 4}

    # Prepare input data
    input_data = np.array([
        neighbourhood_group_map[neighbourhood_group], 
        latitude,
        longitude,
        room_type_map[room_type], 
        minimum_nights, 
        number_of_reviews,
        reviews_per_month,
        calculated_host_listings_count,
        availability_365
    ]).reshape(1, -1)

    # Load model
    try:
        with open("best_price_model.pkl", "rb") as file:
            model = pickle.load(file)

        # Predict price
        if st.button("Predict Price"):
            prediction = model.predict(input_data)[0]
            st.success(f"💰 Estimated Price: ${prediction:.2f} per night")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'best_price_model.pkl' is in the same directory.")
    except ValueError as e:
        st.error(f"Feature shape mismatch: {e}")

if __name__ == "__main__":
    main()


