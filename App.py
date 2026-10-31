
import pickle
from pathlib import Path

from PIL import Image
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

LOGO_PATH = BASE_DIR / "logo.png"
LOGO2_PATH = BASE_DIR / "image.png"
DATA_PATH = BASE_DIR / "Food_Delivery_Times.csv"
MODEL_PATH = BASE_DIR / "final_rf_model.pkl"
ENCODERS_PATH = BASE_DIR / "label_encoders.pkl"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon=Image.open(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)


CATEGORICAL_COLUMNS = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]

FEATURE_COLUMNS = [
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]


# ---------------------------------------------------------
# Load project files
# ---------------------------------------------------------
@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}"
        )

    data = pd.read_csv(DATA_PATH)

    for column in [
        "Weather",
        "Traffic_Level",
        "Time_of_Day"
    ]:

        if column in data.columns and data[column].isna().any():
            data[column] = data[column].fillna(
                data[column].mode().iloc[0]
            )

    if "Courier_Experience_yrs" in data.columns:

        data["Courier_Experience_yrs"] = data[
            "Courier_Experience_yrs"
        ].fillna(
            data["Courier_Experience_yrs"].median()
        )

    return data


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_saved_encoders():

    if not ENCODERS_PATH.exists():
        return None

    with open(ENCODERS_PATH, "rb") as file:
        return pickle.load(file)


try:

    df = load_dataset()
    model = load_model()
    saved_encoders = load_saved_encoders()

except Exception as error:

    st.error(f"Unable to load the project files: {error}")

    st.info(
        "Keep app.py, logo.png, Food_Delivery_Times.csv, "
        "final_rf_model.pkl and label_encoders.pkl "
        "in the same folder."
    )

    st.stop()


# ---------------------------------------------------------
# Category encoding
# ---------------------------------------------------------
category_maps = {}

for column in CATEGORICAL_COLUMNS:

    category_values = sorted(
        df[column].dropna().astype(str).unique().tolist()
    )

    category_maps[column] = {
        value: index
        for index, value in enumerate(category_values)
    }


# ---------------------------------------------------------
# Sidebar branding
# ---------------------------------------------------------
logo_col, title_col = st.sidebar.columns([1, 3])

with logo_col:
    st.image(str(LOGO2_PATH), width=55)

with title_col:
    st.markdown(
        """
        <div style="padding-top: 5px;">
            <h3 style="margin: 0; font-size: 20px;">
                Food Delivery
            </h3>
            <p style="font-size: 11px; color: gray; margin: 0;">
                Time Prediction System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Predict Delivery Time",
        "Data Analysis",
        "Dataset",
        "Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning Project | Random Forest Regressor"
)


# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------
if page == "Home":

    # Main heading with logo
    logo_col, title_col = st.columns([0.8, 9])

    with logo_col:
        st.image(str(LOGO2_PATH), width=75)

    with title_col:
        st.markdown(
            """
            <h1 style="margin-top: 8px; font-size: 34px;">
                Food Delivery Time Prediction
            </h1>
            """,
            unsafe_allow_html=True
        )

    st.write(
        "Estimate how long a food order may take to reach a customer "
        "using distance, weather, traffic, time of day, vehicle type, "
        "preparation time and courier experience."
    )

    total_orders = len(df)

    average_delivery = df["Delivery_Time_min"].mean()

    fastest_delivery = df["Delivery_Time_min"].min()

    longest_delivery = df["Delivery_Time_min"].max()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Orders", f"{total_orders:,}")

    c2.metric(
        "Average Delivery",
        f"{average_delivery:.1f} min"
    )

    c3.metric(
        "Fastest Delivery",
        f"{fastest_delivery:.0f} min"
    )

    c4.metric(
        "Longest Delivery",
        f"{longest_delivery:.0f} min"
    )

    st.subheader("Delivery time distribution")

    fig = px.histogram(
        df,
        x="Delivery_Time_min",
        nbins=30,
        labels={
            "Delivery_Time_min": "Delivery time (minutes)",
            "count": "Orders"
        },
        color_discrete_sequence=["#3987D6"],
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Delivery time (minutes)",
        yaxis_title="Number of orders",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("How the prediction works")

    st.write(
        "The trained Random Forest model uses seven input features "
        "to estimate delivery time. Open **Predict Delivery Time** "
        "in the sidebar to try it."
    )


# ---------------------------------------------------------
# Prediction page
# ---------------------------------------------------------
elif page == "Predict Delivery Time":

    st.title("Predict Delivery Time")

    st.write(
        "Enter the order and courier details below."
    )

    with st.form("prediction_form"):

        left, right = st.columns(2)

        with left:

            distance = st.number_input(
                "Delivery distance (km)",
                min_value=0.1,
                max_value=100.0,
                value=5.0,
                step=0.1,
            )

            weather = st.selectbox(
                "Weather",
                options=list(
                    category_maps["Weather"].keys()
                ),
            )

            traffic = st.selectbox(
                "Traffic level",
                options=list(
                    category_maps["Traffic_Level"].keys()
                ),
            )

            time_of_day = st.selectbox(
                "Time of day",
                options=list(
                    category_maps["Time_of_Day"].keys()
                ),
            )

        with right:

            vehicle = st.selectbox(
                "Vehicle type",
                options=list(
                    category_maps["Vehicle_Type"].keys()
                ),
            )

            preparation = st.number_input(
                "Food preparation time (minutes)",
                min_value=1,
                max_value=180,
                value=15,
                step=1,
            )

            experience = st.number_input(
                "Courier experience (years)",
                min_value=0.0,
                max_value=50.0,
                value=2.0,
                step=0.5,
            )

        submitted = st.form_submit_button(
            "Predict Delivery Time",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame(
            [[
                float(distance),
                category_maps["Weather"][weather],
                category_maps["Traffic_Level"][traffic],
                category_maps["Time_of_Day"][time_of_day],
                category_maps["Vehicle_Type"][vehicle],
                int(preparation),
                float(experience),
            ]],
            columns=FEATURE_COLUMNS,
        )

        try:

            prediction = float(
                model.predict(input_data)[0]
            )

            prediction = max(0.0, prediction)

            st.success("Prediction completed!")

            st.metric(
                "Estimated delivery time",
                f"{prediction:.1f} minutes"
            )

            st.caption(
                "This is a model estimate, not a guaranteed delivery time. "
                "Actual delivery may vary."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

            st.write(
                "Check that the model and encoder files are from "
                "the same training run as this project."
            )


# ---------------------------------------------------------
# Data analysis page
# ---------------------------------------------------------
elif page == "Data Analysis":

    st.title("Data Analysis")

    st.write(
        "Explore patterns and relationships in the food delivery dataset."
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="Distance_km",
            y="Delivery_Time_min",
            color="Traffic_Level",
            title="Distance vs Delivery Time",
            labels={
                "Distance_km": "Distance (km)",
                "Delivery_Time_min": "Delivery time (minutes)",
            },
            template="plotly_white",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        average_by_weather = (
            df.groupby(
                "Weather",
                as_index=False
            )["Delivery_Time_min"].mean()
        )

        fig = px.bar(
            average_by_weather,
            x="Weather",
            y="Delivery_Time_min",
            title="Average Delivery Time by Weather",
            labels={
                "Delivery_Time_min": "Average delivery time (minutes)"
            },
            template="plotly_white",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    with col3:

        average_by_traffic = (
            df.groupby(
                "Traffic_Level",
                as_index=False
            )["Delivery_Time_min"].mean()
        )

        fig = px.bar(
            average_by_traffic,
            x="Traffic_Level",
            y="Delivery_Time_min",
            title="Average Delivery Time by Traffic",
            labels={
                "Delivery_Time_min": "Average delivery time (minutes)"
            },
            template="plotly_white",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col4:

        fig = px.box(
            df,
            x="Vehicle_Type",
            y="Delivery_Time_min",
            color="Vehicle_Type",
            title="Delivery Time by Vehicle Type",
            labels={
                "Delivery_Time_min": "Delivery time (minutes)"
            },
            template="plotly_white",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Numerical correlation")

    numeric_columns = [
        "Distance_km",
        "Preparation_Time_min",
        "Courier_Experience_yrs",
        "Delivery_Time_min",
    ]

    corr = df[numeric_columns].corr(
        numeric_only=True
    )

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Descriptive statistics")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )


# ---------------------------------------------------------
# Dataset page
# ---------------------------------------------------------
elif page == "Dataset":

    st.title("Food Delivery Dataset")

    st.write(
        f"Dataset contains **{len(df):,} rows** "
        f"and **{len(df.columns)} columns**."
    )

    search_text = st.text_input(
        "Search the dataset",
        placeholder="Enter a value or keyword"
    )

    display_df = df.copy()

    if search_text:

        mask = display_df.astype(str).apply(
            lambda column: column.str.contains(
                search_text,
                case=False,
                na=False
            )
        ).any(axis=1)

        display_df = display_df[mask]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="food_delivery_data.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# Model information page
# ---------------------------------------------------------
elif page == "Model Information":

    st.title("Model Information")

    st.write(
        "The project uses a tuned Random Forest Regressor."
    )

    a, b, c = st.columns(3)

    a.metric(
        "R² Score",
        "0.7826"
    )

    b.metric(
        "Mean Absolute Error",
        "7.2575 min"
    )

    c.metric(
        "Root Mean Squared Error",
        "9.8719 min"
    )

    st.subheader("Model configuration")

    parameters = {
        "Algorithm": "RandomForestRegressor",
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "log2",
        "random_state": 42,
    }

    st.table(
        pd.DataFrame(
            parameters.items(),
            columns=["Parameter", "Value"]
        )
    )

    st.subheader("Features used by the model")

    st.write(
        "Distance_km, Weather, Traffic_Level, Time_of_Day, "
        "Vehicle_Type, Preparation_Time_min and "
        "Courier_Experience_yrs."
    )

    st.info(
        "Evaluation metrics are copied from the final model evaluation "
        "in your uploaded notebook. They describe the test-set results "
        "reported during training."
    )


# ---------------------------------------------------------
# Sidebar footer
# ---------------------------------------------------------
st.sidebar.divider()

st.sidebar.caption(
    "Built with Streamlit and Scikit-learn"
)
