import streamlit as st
import joblib
import os
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Food Delivery ML",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS — Colorful Theme
# ============================================================

st.markdown("""
<style>

/* ========================================================
   GLOBAL
   ======================================================== */

.stApp {
    background: #fbfaff;
}

.main .block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* ========================================================
   SIDEBAR
   ======================================================== */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16a34a 0%, #22c55e 50%, #65a30d 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 8px 10px;
    border-radius: 8px;
}

/* ========================================================
   HEADINGS
   ======================================================== */

h1 {
    color: #14532d !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

h2 {
    color: #15803d !important;
    font-weight: 700 !important;
}

h3 {
    color: #16a34a !important;
    font-weight: 700 !important;
}

/* ========================================================
   METRIC CARDS
   ======================================================== */

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dcfce7;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(22, 163, 74, 0.10);
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #16a34a !important;
    font-weight: 800;
}

/* ========================================================
   INPUTS
   ======================================================== */

.stSelectbox,
.stNumberInput {
    margin-bottom: 8px;
}

/* Force field labels (e.g. "Cuisine Type", "Restaurant Rating") to
   always be visible, regardless of browser/system dark mode */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] {
    color: #14532d !important;
    font-weight: 600 !important;
    opacity: 1 !important;
    font-size: 14px !important;
}

div[data-baseweb="select"] {
    border-radius: 10px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="select"] div,
div[data-baseweb="select"] > div > div,
div[data-baseweb="select"] [class] {
    background-color: #ffffff !important;
    border-color: #dcfce7 !important;
    color: #14532d !important;
}

div[data-baseweb="select"] * {
    color: #14532d !important;
    fill: #14532d !important;
}

div[data-baseweb="select"] svg {
    color: #14532d !important;
}

div[data-testid="stNumberInput"] input {
    border-radius: 10px;
    background: white !important;
    color: #14532d !important;
    border: 1px solid #dcfce7 !important;
}

div[data-testid="stNumberInput"] button {
    background: #f0fdf4 !important;
    color: #14532d !important;
    border: 1px solid #dcfce7 !important;
}

/* Dropdown option list when a selectbox is opened */
ul[data-testid="stSelectboxVirtualDropdown"] {
    background: white !important;
}

ul[data-testid="stSelectboxVirtualDropdown"] li {
    color: #14532d !important;
    background: white !important;
}

ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
    background: #f0fdf4 !important;
}

/* ========================================================
   BUTTONS
   ======================================================== */

.stButton > button {
    border-radius: 12px;
    min-height: 50px;
    font-size: 16px;
    font-weight: 700;
    border: none;
    color: white;
    background: linear-gradient(135deg, #16a34a, #65a30d, #0d9488);
    background-size: 200% 200%;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(22, 163, 74, 0.35);
    background-position: 100% 50%;
}

/* ========================================================
   CUSTOM CARDS
   ======================================================== */

.hero-card {
    background: linear-gradient(135deg, #16a34a, #22c55e 45%, #65a30d 80%, #0d9488);
    padding: 38px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(22, 163, 74, 0.25);
}

.hero-card h1 {
    color: white !important;
    font-size: 42px;
    margin-bottom: 10px;
}

.hero-card p {
    color: #ecfdf5;
    font-size: 18px;
    line-height: 1.6;
}

.section-card {
    background: white;
    border: 1px solid #dcfce7;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 5px 18px rgba(22, 163, 74, 0.06);
}

.prediction-card {
    background: linear-gradient(135deg, #ecfdf5, #f0fdf4, #fefce8);
    border: 2px solid #86efac;
    border-radius: 22px;
    padding: 35px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 10px 30px rgba(22, 163, 74, 0.15);
}

.prediction-label {
    color: #16a34a;
    font-size: 17px;
    font-weight: 700;
}

.prediction-value {
    background: linear-gradient(135deg, #16a34a, #65a30d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 46px;
    font-weight: 900;
    margin: 10px 0;
}

.prediction-subtitle {
    color: #64748b;
    font-size: 15px;
}

.feature-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #dcfce7;
    height: 100%;
    box-shadow: 0 5px 18px rgba(22, 163, 74, 0.06);
}

.feature-icon {
    font-size: 30px;
}

.feature-title {
    font-size: 18px;
    font-weight: 700;
    color: #15803d;
    margin-top: 8px;
}

.feature-text {
    color: #64748b;
    font-size: 14px;
    line-height: 1.5;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background: linear-gradient(135deg, #dcfce7, #d9f99d);
    color: #16a34a;
    font-size: 13px;
    font-weight: 700;
}

.footer {
    text-align: center;
    padding: 30px 10px 10px;
    color: #94a3b8;
    font-size: 13px;
}

/* ========================================================
   DATAFRAME
   ======================================================== */

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* ========================================================
   INFO / SUCCESS / WARNING
   ======================================================== */

div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "..",
        "models",
        "food_delivery_time_random_forest.pkl"
    )
)

print("BASE_DIR:", BASE_DIR)
print("MODEL PATH:", model_path)
print("MODEL EXISTS:", os.path.isfile(model_path))

if not os.path.isfile(model_path):
    st.error("❌ Model file was not found!")
    st.write("Expected path:", model_path)
    st.stop()

try:
    model = joblib.load(model_path)
    print("Model loaded successfully")

except Exception as e:
    st.error("❌ Model loading failed")
    st.exception(e)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
<div style="text-align:center; padding:15px 5px 20px;">
<div style="font-size:42px;">🍔</div>
<h2 style="color:white !important; margin:0; font-size:22px;">Food Delivery ML</h2>
<p style="color:#ecfdf5; font-size:12px; margin-top:5px;">Machine Learning Dashboard</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<p style='font-size:12px; font-weight:700;'>NAVIGATION</p>", unsafe_allow_html=True)

    page = st.radio(
        "",
        [
            "🏠 Home",
            "🔮 Prediction",
            "📊 Model Performance",
            "🔍 Data Insights",
            "🔵 K-Means",
            "🟢 DBSCAN",
            "🔗 Apriori"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
<div style="text-align:center; color:#ecfdf5; font-size:12px; line-height:1.6;">
<b>ML Project</b><br>
Food Delivery Time Prediction<br>
<span style="font-size:11px;">Random Forest Regression</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown("""
<div class="hero-card">
<div class="badge" style="background:rgba(255,255,255,0.2); color:white;">MACHINE LEARNING PROJECT</div>
<h1>🍔 Food Delivery Time Prediction</h1>
<p>Predict food delivery time using machine learning based on order, restaurant, rider, traffic, distance and delivery information.</p>
</div>
""", unsafe_allow_html=True)

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.subheader("📊 Model at a Glance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("R² Score", "99.09%", "Excellent")

    with c2:
        st.metric("MAE", "2.63 min", "Low Error")

    with c3:
        st.metric("RMSE", "3.40 min", "Low Error")

    with c4:
        st.metric("Best Model", "Random Forest", "Tuned")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    st.subheader("🤖 Project Overview")

    col1, col2 = st.columns([1.3, 1])

    with col1:

        st.markdown("""
<div class="section-card">
<h3>🎯 What does this project do?</h3>
<p style="color:#64748b; line-height:1.7;">This project uses Machine Learning to estimate how long a food delivery order will take. Multiple regression algorithms were trained and compared to identify the best-performing model.</p>
<h3>🧠 Machine Learning Pipeline</h3>
<p style="color:#64748b; line-height:1.8;">📥 Data Collection → 🧹 Data Cleaning → 📊 EDA → ⚙️ Preprocessing → 🤖 Model Training → 🎯 Hyperparameter Tuning → 🚀 Prediction</p>
</div>
""", unsafe_allow_html=True)

    with col2:

        st.markdown("""
<div class="section-card">
<h3>🏆 Best Algorithm</h3>
<div style="font-size:28px; font-weight:800; background:linear-gradient(135deg,#16a34a,#65a30d); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin:15px 0;">🌲 Tuned Random Forest</div>
<p style="color:#64748b;">The tuned Random Forest model achieved the highest R² score and the lowest prediction error.</p>
<hr>
<p><b>R²:</b> 99.09%<br><b>MAE:</b> 2.63 minutes<br><b>RMSE:</b> 3.40 minutes</p>
</div>
""", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PROJECT MODULES
    # --------------------------------------------------------

    st.subheader("🚀 Explore the Project")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
<div class="feature-card">
<div class="feature-icon">🔮</div>
<div class="feature-title">Delivery Prediction</div>
<div class="feature-text">Enter order and delivery information to predict delivery time.</div>
</div>
""", unsafe_allow_html=True)

    with f2:
        st.markdown("""
<div class="feature-card">
<div class="feature-icon">📊</div>
<div class="feature-title">Model Analysis</div>
<div class="feature-text">Compare regression algorithms and analyze model performance.</div>
</div>
""", unsafe_allow_html=True)

    with f3:
        st.markdown("""
<div class="feature-card">
<div class="feature-icon">🔍</div>
<div class="feature-title">Data Mining</div>
<div class="feature-text">Explore K-Means, DBSCAN and Apriori analysis results.</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="footer">🍔 Food Delivery Time Prediction | Machine Learning Project | Tuned Random Forest</div>
""", unsafe_allow_html=True)


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔮 Prediction":

    st.title("🔮 Delivery Time Predictor")

    st.markdown("""
Predict the estimated delivery time by entering the order, restaurant, rider, traffic and distance details.
""")

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    st.markdown("""
<div class="section-card">
<span class="badge">● MODEL READY</span>
<h3>🌲 Tuned Random Forest Regression</h3>
<p style="color:#64748b;">The prediction engine is trained and ready to estimate food delivery time.</p>
</div>
""", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.subheader("📊 Model Performance")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("R² Score", "99.09%")

    with m2:
        st.metric("MAE", "2.63 min")

    with m3:
        st.metric("RMSE", "3.40 min")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # ORDER INFORMATION
    # --------------------------------------------------------

    st.markdown("""
<div class="section-card"><h3>📦 Order Information</h3></div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        order_hour = st.number_input(
            "🕐 Order Hour",
            min_value=0,
            max_value=23,
            value=12
        )

    with c2:
        day_of_week = st.selectbox(
            "📅 Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

    with c3:
        is_weekend = st.selectbox(
            "📆 Weekend?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with c4:
        is_festival = st.selectbox(
            "🎉 Festival?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    # --------------------------------------------------------
    # RESTAURANT INFORMATION
    # --------------------------------------------------------

    st.markdown("""
<div class="section-card"><h3>🏪 Restaurant Information</h3></div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        cuisine_type = st.selectbox(
            "🍽️ Cuisine Type",
            [
                "Pizza",
                "Desserts",
                "Biryani",
                "Burger",
                "South Indian",
                "Bakery",
                "North Indian",
                "Cafe",
                "Chinese"
            ]
        )

    with c2:
        restaurant_rating = st.number_input(
            "⭐ Restaurant Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.2,
            step=0.1
        )

    with c3:
        restaurant_load = st.selectbox(
            "🏪 Restaurant Load",
            ["Low", "Medium", "High"]
        )

    with c4:
        order_items = st.number_input(
            "📦 Order Items",
            min_value=1,
            max_value=20,
            value=2
        )

    # --------------------------------------------------------
    # RIDER INFORMATION
    # --------------------------------------------------------

    st.markdown("""
<div class="section-card"><h3>🛵 Rider Information</h3></div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        vehicle_type = st.selectbox(
            "🛵 Vehicle Type",
            [
                "Scooter",
                "Bike",
                "Electric Scooter",
                "Bicycle"
            ]
        )

    with c2:
        rider_experience = st.number_input(
            "👨‍🚴 Experience (Years)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.5
        )

    with c3:
        rider_rating = st.number_input(
            "⭐ Rider Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.5,
            step=0.1
        )

    # --------------------------------------------------------
    # DELIVERY INFORMATION
    # --------------------------------------------------------

    st.markdown("""
<div class="section-card"><h3>🚚 Delivery & Traffic Information</h3></div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        pickup_zone = st.selectbox(
            "📍 Pickup Zone",
            [
                "CBD",
                "Residential",
                "Industrial",
                "Commercial",
                "Suburban"
            ]
        )

    with c2:
        dropoff_zone = st.selectbox(
            "📍 Dropoff Zone",
            [
                "Commercial",
                "CBD",
                "Residential",
                "Industrial",
                "Suburban"
            ]
        )

    with c3:
        weather = st.selectbox(
            "🌦️ Weather",
            [
                "Clear",
                "Rain",
                "Cloudy",
                "Storm",
                "Fog"
            ]
        )

    with c4:
        traffic_level = st.selectbox(
            "🚦 Traffic Level",
            [
                "Low",
                "Moderate",
                "High",
                "Severe"
            ]
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        preparation_time = st.number_input(
            "👨‍🍳 Preparation Time (min)",
            min_value=1,
            max_value=120,
            value=20
        )

    with c2:
        road_distance = st.number_input(
            "🛣️ Road Distance (km)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1
        )

    with c3:
        delivery_distance = st.selectbox(
            "📏 Distance Category",
            [
                "Short",
                "Medium",
                "Long"
            ]
        )

    with c4:
        number_of_signals = st.number_input(
            "🚦 Number of Signals",
            min_value=0,
            max_value=50,
            value=5
        )

    c1, c2 = st.columns(2)

    with c1:
        average_speed = st.number_input(
            "🏎️ Average Speed (km/h)",
            min_value=1.0,
            max_value=100.0,
            value=35.0,
            step=0.5
        )

    with c2:
        delivery_priority = st.selectbox(
            "⚡ Delivery Priority",
            [
                "Normal",
                "VIP",
                "Priority"
            ]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮  PREDICT DELIVERY TIME",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        input_data = pd.DataFrame(
            [{
                "Order_Hour": order_hour,
                "Day_of_Week": day_of_week,
                "Is_Weekend": is_weekend,
                "Is_Festival": is_festival,
                "Weather": weather,
                "Pickup_Zone": pickup_zone,
                "Dropoff_Zone": dropoff_zone,
                "Vehicle_Type": vehicle_type,
                "Rider_Experience_Years": rider_experience,
                "Rider_Rating": rider_rating,
                "Restaurant_Rating": restaurant_rating,
                "Cuisine_Type": cuisine_type,
                "Order_Items": order_items,
                "Restaurant_Load": restaurant_load,
                "Preparation_Time_Min": preparation_time,
                "Road_Distance_km": road_distance,
                "Delivery_Distance_Category": delivery_distance,
                "Traffic_Level": traffic_level,
                "Number_of_Signals": number_of_signals,
                "Average_Speed_kmph": average_speed,
                "Delivery_Priority": delivery_priority
            }]
        )

        prediction = model.predict(input_data)

        predicted_time = prediction[0]

        # ------------------------------------------------
        # PREDICTION RESULT
        # ------------------------------------------------

        st.markdown(f"""
<div class="prediction-card">
<div class="prediction-label">🎯 ESTIMATED DELIVERY TIME</div>
<div class="prediction-value">{float(predicted_time):.2f} Minutes</div>
<div class="prediction-subtitle">Prediction generated by the Tuned Random Forest model</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.markdown("""
Compare the regression algorithms used in this project and understand why Tuned Random Forest was selected.
""")

    results = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Tuned Random Forest"
            ],
            "MAE": [
                6.348329,
                3.866554,
                2.626114,
                2.625041
            ],
            "RMSE": [
                9.229774,
                5.111817,
                3.397755,
                3.395733
            ],
            "R² Score": [
                0.932935,
                0.979428,
                0.990911,
                0.990922
            ]
        }
    )

    # --------------------------------------------------------
    # BEST MODEL CARDS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Best R² Score", "99.09%")

    with c2:
        st.metric("Lowest MAE", "2.63 min")

    with c3:
        st.metric("Lowest RMSE", "3.40 min")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader("📋 Model Comparison")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader("📈 R² Score Comparison")

    chart_data = results.set_index("Model")["R² Score"]

    st.bar_chart(chart_data)

    # --------------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------------

    st.markdown("""
<div class="section-card">
<h3>🏆 Final Model Selection</h3>
<p style="color:#64748b; font-size:16px; line-height:1.7;">The <b>Tuned Random Forest</b> model achieved the best performance with an R² score of <b>99.09%</b>, MAE of <b>2.63 minutes</b>, and RMSE of <b>3.40 minutes</b>.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DATA INSIGHTS PAGE
# ============================================================

elif page == "🔍 Data Insights":

    st.title("🔍 Data Insights")

    st.markdown("""
Explore the features that contribute most strongly to the Random Forest model's predictions.
""")

    feature_names = [
        "Road Distance",
        "Average Speed",
        "Preparation Time",
        "Number of Signals",
        "Order Hour"
    ]

    importance_values = [
        0.510074,
        0.426976,
        0.054468,
        0.001857,
        0.000886
    ]

    feature_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance_values
        }
    )

    # --------------------------------------------------------
    # TOP FEATURES
    # --------------------------------------------------------

    st.subheader("🎯 Most Important Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "🥇 Road Distance",
            "51.01%"
        )

    with c2:
        st.metric(
            "🥈 Average Speed",
            "42.70%"
        )

    with c3:
        st.metric(
            "🥉 Preparation Time",
            "5.45%"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader("📊 Feature Importance")

    st.bar_chart(
        feature_importance.set_index("Feature")
    )

    # --------------------------------------------------------
    # FINDING
    # --------------------------------------------------------

    st.markdown("""
<div class="section-card">
<h3>💡 Key Finding</h3>
<p style="color:#64748b; line-height:1.7;"><b>Road Distance</b> and <b>Average Speed</b> are the two dominant features in the Random Forest feature-importance analysis. <b>Preparation Time</b> is the third most important feature.</p>
<p style="color:#64748b; font-size:13px;">Note: Feature importance indicates how the trained model uses these features. It does not by itself establish causation.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# K-MEANS PAGE
# ============================================================

elif page == "🔵 K-Means":

    st.title("🔵 K-Means Clustering")

    st.markdown("""
K-Means groups delivery orders into four profiles based on rider, restaurant, preparation, distance, traffic and speed characteristics.
""")

    cluster_data = pd.DataFrame(
        {
            "Cluster": [
                "Cluster 0",
                "Cluster 1",
                "Cluster 2",
                "Cluster 3"
            ],
            "Orders": [
                9442,
                18151,
                10503,
                11904
            ],
            "Average Delivery Time (min)": [
                77.78,
                75.66,
                88.00,
                97.60
            ],
            "Profile": [
                "Experienced & faster riders",
                "Quick & smaller orders",
                "Large & time-consuming orders",
                "Slow / high-signal deliveries"
            ]
        }
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Cluster 0", "9,442")

    with c2:
        st.metric("Cluster 1", "18,151")

    with c3:
        st.metric("Cluster 2", "10,503")

    with c4:
        st.metric("Cluster 3", "11,904")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader("📋 Cluster Summary")

    st.dataframe(
        cluster_data,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader("⏱️ Average Delivery Time")

    st.bar_chart(
        cluster_data.set_index("Cluster")[
            "Average Delivery Time (min)"
        ]
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader("💡 Cluster Interpretation")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
<div class="feature-card">
<h3>🔵 Cluster 0</h3>
<p class="feature-text">Experienced and faster riders with an average delivery time of approximately 77.78 minutes.</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
<div class="feature-card">
<h3>🟢 Cluster 1</h3>
<p class="feature-text">Quick deliveries with smaller orders. This cluster has the lowest average delivery time at approximately 75.66 minutes.</p>
</div>
""", unsafe_allow_html=True)

    with c2:

        st.markdown("""
<div class="feature-card">
<h3>🟡 Cluster 2</h3>
<p class="feature-text">Larger and more time-consuming orders with an average delivery time of approximately 88 minutes.</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
<div class="feature-card">
<h3>🔴 Cluster 3</h3>
<p class="feature-text">Slower and high-signal deliveries. This cluster has the highest average delivery time at approximately 97.60 minutes.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DBSCAN PAGE
# ============================================================

elif page == "🟢 DBSCAN":

    st.title("🟢 DBSCAN Clustering")

    st.markdown("""
DBSCAN is a density-based clustering algorithm used to identify dense groups and unusual/noisy observations.
""")

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    st.subheader("⚙️ DBSCAN Configuration")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Epsilon (eps)", "1.6")

    with c2:
        st.metric("Minimum Samples", "5")

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    dbscan_data = pd.DataFrame(
        {
            "Cluster": [
                "-1 (Noise)",
                "0",
                "1"
            ],
            "Number of Points": [
                44,
                49953,
                3
            ]
        }
    )

    st.subheader("📊 DBSCAN Results")

    st.dataframe(
        dbscan_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 Cluster Distribution")

    st.bar_chart(
        dbscan_data.set_index("Cluster")
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.markdown("""
<div class="section-card">
<h3>💡 Interpretation</h3>
<p style="color:#64748b; line-height:1.7;">DBSCAN produced one very large cluster containing most of the observations, a very small second cluster, and <b>44 noise points</b>.</p>
<p style="color:#64748b; line-height:1.7;">This indicates that DBSCAN did not reveal strong separate density-based groups in this dataset.</p>
</div>
""", unsafe_allow_html=True)

    st.warning(
        "For this dataset, K-Means provides more interpretable "
        "and useful delivery profiles than DBSCAN."
    )


# ============================================================
# APRIORI PAGE
# ============================================================

elif page == "🔗 Apriori":

    st.title("🔗 Apriori Association Analysis")

    st.markdown("""
Apriori was used to discover relationships between delivery attributes such as weather, traffic, vehicle type, priority, cuisine, restaurant load and delivery distance.
""")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Frequent Itemsets",
            "143"
        )

    with c2:
        st.metric(
            "Association Rules",
            "167"
        )

    with c3:
        st.metric(
            "Best Lift",
            "1.17"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # BEST RULE
    # --------------------------------------------------------

    st.markdown("""
<div class="prediction-card">
<div class="prediction-label">🏆 STRONGEST EXAMPLE ASSOCIATION</div>
<div style="background:linear-gradient(135deg,#16a34a,#65a30d); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; font-size:28px; font-weight:800; margin:15px;">Clear Weather → Low Traffic</div>
<div class="prediction-subtitle">Positive association between clear weather and low traffic.</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # RULE METRICS
    # --------------------------------------------------------

    rule_data = pd.DataFrame(
        {
            "Metric": [
                "Support",
                "Confidence",
                "Lift"
            ],
            "Value": [
                "36.44%",
                "75.93%",
                "1.17"
            ]
        }
    )

    st.subheader("📐 Rule Metrics")

    st.dataframe(
        rule_data,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # TOP RULES
    # --------------------------------------------------------

    st.subheader("📈 Top Association Rules")

    rules_data = pd.DataFrame(
        {
            "Rule": [
                "Clear → Low Traffic",
                "Long Distance → Normal Priority",
                "Low Traffic → Long Distance",
                "Clear Weather → Long Distance",
                "Low Traffic → Normal Priority"
            ],
            "Lift": [
                1.17,
                1.11,
                1.09,
                1.06,
                1.04
            ]
        }
    )

    st.dataframe(
        rules_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("💡 Interpretation")

    st.info(
        """
        The rule **Clear Weather → Low Traffic** has a confidence
        of 75.93%. This means that among clear-weather orders,
        approximately 75.93% are associated with low traffic.

        A lift of 1.17 indicates a positive association.

        ⚠️ Association does not mean causation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
<hr style="border:0;border-top:1px solid #dcfce7;">
<b>🍔 Food Delivery Time Prediction</b><br>
Machine Learning & Data Mining Project<br>
Tuned Random Forest • K-Means • DBSCAN • Apriori
</div>
""", unsafe_allow_html=True)