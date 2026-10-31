# Food Delivery Time Prediction

## Project Overview

This project focuses on predicting food delivery time using Machine Learning. The model estimates how much time a food order may take to reach the customer based on different factors such as delivery distance, weather conditions, traffic level, time of day, vehicle type, food preparation time, and courier experience.

A Random Forest Regressor is used to train the prediction model. I have also developed a Streamlit web application where users can enter delivery details and get an estimated delivery time.

## Project Objectives

- Predict food delivery time using historical delivery data.
- Understand how different factors affect delivery duration.
- Analyze the impact of distance, traffic, weather, and vehicle type.
- Build a web application to make predictions easily.
- Visualize useful information from the dataset.

## Dataset Information

The project uses the `Food_Delivery_Times.csv` dataset.

The following input features are used to predict delivery time:

| Feature | Description |
|---|---|
| Distance_km | Delivery distance in kilometers |
| Weather | Weather conditions |
| Traffic_Level | Traffic conditions |
| Time_of_Day | Time of the order |
| Vehicle_Type | Type of delivery vehicle |
| Preparation_Time_min | Food preparation time |
| Courier_Experience_yrs | Courier experience in years |

**Target Variable:** `Delivery_Time_min`

## Machine Learning Model

I used the Random Forest Regressor algorithm for this project.

The model development process includes:

1. Loading and understanding the dataset.
2. Handling missing values.
3. Encoding categorical features.
4. Splitting the data into training and testing sets.
5. Training the Random Forest Regressor.
6. Performing hyperparameter tuning.
7. Evaluating the model using regression metrics.
8. Saving the trained model for use in the Streamlit application.

## Model Performance

The final model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| MAE | 7.2575 |
| MSE | 97.4546 |
| RMSE | 9.8719 |
| R² Score | 0.7826 |

The R² score of 0.7826 indicates that the model explains approximately 78.26% of the variation in delivery time in the test dataset.

## Streamlit Application

The project includes a Streamlit application with the following sections:

- **Overview:** Displays general information and statistics about the dataset.
- **Predict Delivery:** Allows users to enter delivery details and predict the estimated delivery time.
- **Analytics:** Displays visualizations related to delivery distance, weather, traffic, and vehicle types.
- **Data Explorer:** Allows users to explore and filter the dataset.
- **Model Performance:** Displays evaluation metrics and feature importance.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

```text
Food_Delivery_Time_Project/
│
├── app.py
├── model.ipynb
├── Food_Delivery_Times.csv
├── final_rf_model.pkl
├── label_encoders.pkl
├── requirements.txt
└── README.md
```

## How to Run the Project

**1. Clone the repository**

```bash
git clone https://github.com/sanskargadhe/Food_Delivery_Time_Prediction-.git
```

**2. Navigate to the project directory**

```bash
cd Food_Delivery_Time_Project
```

**3. Install the required libraries**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

The application will open in your browser at:

`http://localhost:8501`

## Important Note About Encoders

The supplied `label_encoders.pkl` file stores integer classes. To maintain consistency with the original model training process, the application recreates the alphabetical category mappings using the original CSV dataset.

Make sure the dataset and trained model files are available in the project directory before running the application.

## Future Improvements

- Improve prediction performance through feature engineering.
- Experiment with other machine learning algorithms.
- Add real-time weather and traffic information.
- Deploy the application online.

## Author

**Sanskar Gadhe**

B.Tech – Electronics and Communication Engineering

GitHub: https://github.com/sanskargadhe
