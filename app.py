import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="KSA Real Estate Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base_path, 'real_estate_model.pkl'))
    features = joblib.load(os.path.join(base_path, 'model_features.pkl'))
    return model, features

try:
    model, features = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# قاموس إحداثيات المناطق
provinces_coords = {
    "Riyadh": [24.7136, 46.6753],
    "Makkah": [21.3891, 39.8579],
    "Madinah": [24.4672, 39.6024],
    "Eastern": [26.4207, 50.0888],
    "Qasim": [26.3273, 43.9784],
    "Aseer": [18.2171, 42.5053],
    "Tabuk": [28.3835, 36.5662],
    "Hail": [27.5114, 41.7208],
    "Northern Borders": [30.9753, 41.0381],
    "Jazan": [16.8892, 42.5511],
    "Najran": [17.4933, 44.1322],
    "Al-Bahah": [20.0129, 41.4677],
    "Al-Jawf": [29.9697, 40.2064]
}

st.sidebar.header("📊 Model Constraints")
st.sidebar.markdown(f"""
The model is optimized for:
- Max Area: 688 m²
- **Bedrooms: 1 to 6
- **Property Age: Up to 2.5 years
""")

st.sidebar.markdown("---")
st.sidebar.markdown(" 🚀 Powered by:")
st.sidebar.info("- Fady Talat - Fatma Ashraf - Ali Kamar - Mohamed Kamal - Eslam Gamal")

st.title("🏠 Saudi Real Estate AI Predictor")
st.markdown("Select the province and property details to get an AI-powered price estimation.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📏 Physical Specs")
    beds = st.number_input("Bedrooms", 2.0, 6.0, 4.0, step=0.5)
    livings = st.number_input("Living Rooms", 0.0, 6.0, 2.0, step=1.0)
    wc = st.number_input("Bathrooms", 1.0, 5.0, 4.0, step=1.0)
    area = st.number_input("Total Area (m²)", 1.0, 688.0, 327.0)

with col2:
    st.subheader("🏗️ Attributes")
    street_width = st.number_input("Street Width (m)", 3.0, 35.0, 20.0)
    age = st.number_input("Age (Years)", 0.0, 2.5, 0.5, step=0.1)
    
    kitchen = st.selectbox("Has Kitchen?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_new = st.selectbox("Is Property New?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

with col3:
    st.subheader("📍 Location & Type")
    # قائمة اختيار المنطقة
    selected_province = st.selectbox("Select Province", list(provinces_coords.keys()), index=0)
    
    # الحصول على الإحداثيات بناءً على الاختيار
    lat = provinces_coords[selected_province][0]
    lng = provinces_coords[selected_province][1]
    
    type_options = ["Apartment", "Building", "Floor", "House", "Villa"]
    selected_type = st.selectbox("Property Type", type_options, index=4)
    type_map = {"Apartment": 0, "Building": 1, "Floor": 2, "House": 3, "Villa": 4}
    property_category = type_map[selected_type]

    # تحديد ما إذا كانت الرياض تلقائياً بناءً على اختيار المنطقة
    is_riyadh = 1 if selected_province == "Riyadh" else 0
    st.info(f"Location Set: {selected_province}")

st.divider()

if st.button("🚀 Calculate Estimated Market Price", use_container_width=True):
    with st.spinner('Running Random Forest Inference...'):
        
        # ترتيب المدخلات ليتوافق مع الموديل
        input_values = [
            beds, livings, wc, area, street_width, age,
            kitchen, property_category, is_riyadh, is_new, lat, lng
        ]
        
        input_df = pd.DataFrame([input_values], columns=features)
        
        try:
            log_pred = model.predict(input_df)
            final_price = np.expm1(log_pred)[0]
            
            st.balloons()
            st.success(f" Estimated Price: {final_price:,.2f} SAR")
            
            market_mean = 1613389.0
            diff = final_price - market_mean
            
            # عرض تفاصيل الموقع المستخدم في الحساب (اختياري للتأكيد)
            st.write(f"Calculation Basis:** Calculated for {selected_province} (Lat: {lat}, Lng: {lng})")
            
            if diff > 0:
                st.info(f"💡 This price is {abs(diff):,.2f} SAR above the general market average.")
            else:
                st.info(f"💡 This price is {abs(diff):,.2f} SAR below the general market average.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("Real Estate Pricing Engine v2.0 | Accuracy: 90.8%")