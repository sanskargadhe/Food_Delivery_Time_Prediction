# Food Delivery Time Prediction - Streamlit

## Files
- `app.py`: Streamlit application
- `Food_Delivery_Times.csv`: project dataset
- `final_rf_model.pkl`: trained Random Forest model
- `label_encoders.pkl`: saved encoder artifact
- `requirements.txt`: Python dependencies

## Run locally
1. Install Python.
2. Open a terminal in this folder.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start the website:

   `streamlit run app.py`

The app opens in your browser, normally at http://localhost:8501.

## Important note about encoders
The supplied `label_encoders.pkl` stores integer classes. The app recreates the
alphabetical category mappings from the original CSV, matching the encoded
values shown in the uploaded notebook.
