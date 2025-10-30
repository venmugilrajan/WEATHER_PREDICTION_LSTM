# 🌡️ Weather Temperature Forecasting with LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to forecast the next 10-minute temperature reading based on the previous 10 hours (60 data points) of historical weather data.

The project includes:
1.  A Python script to train and save the LSTM model.
2.  An interactive web app built with Gradio to make live predictions.

## 📈 Visualizations

The training script automatically generates plots to show the model's performance:

* **`model_loss_history.png`**: Shows the model's "learning curve." You can see the error (loss) decrease over each epoch, proving the model learned successfully.
    
* **`prediction_vs_actual.png`**: Compares the model's predictions (red dotted line) against the actual temperatures (blue line) from the test dataset.
    

## 🗂️ Files in this Repository

* **`train.py`**: The main script to train the model. It loads `weather.csv`, processes the data, builds the LSTM, trains it, and saves the final model and scaler.
* **`app.py`**: The Gradio web application. It loads the saved model and scaler to let you perform live forecasts by manually entering data.
* **`requirements.txt`**: A list of all necessary Python packages.
* **`weather.csv`**: The raw weather data used for training and testing.
* **`code.ipynb`**: A Jupyter Notebook version of the training process (used for development and testing).

---

## 🚀 How to Use

Follow these steps to train the model and launch the interactive app.

### Step 1: Setup and Installation

1.  **Get the files:** Make sure all the files (`train.py`, `app.py`, `weather.csv`, `requirements.txt`) are in the same directory.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Train the Model

1.  Run the `train.py` script from your terminal:
    ```bash
    python train.py
    ```

2.  This script will:
    * Load `weather.csv`.
    * Build and train the LSTM model (this may take a few minutes).
    * Print the final Training and Testing RMSE scores.
    * Generate the visualization plots (`model_loss_history.png` and `prediction_vs_actual.png`).
    * Save the two most important files for the app:
        * `lstm_weather_model.keras` (the trained model)
        * `weather_scaler.joblib` (the scaler used to process the data)

### Step 3: Run the Gradio Web App

1.  Once the training is complete, run the `app.py` file using Gradio:
    ```bash
    gradio app.py
    ```

2.  Your terminal will show a local URL, usually `http://127.0.0.1:7860`. Open this URL in your web browser.

3.  **How to use the app:**
    * The model requires **exactly 60** previous temperature readings to make one prediction.
    * Find a sequence of 60 temperatures from your `weather.csv` file.
    * Copy and paste them into the input box as a **single line, separated by commas**.
    * Click the "Forecast Next Temperature" button.
    * The app will show the model's prediction for the next 10-minute time step.

    **Example Input (for testing "hot"):**
    ```
    23.45, 23.82, 24.16, 24.47, 24.73, 24.96, 25.17, 25.36, 25.53, 25.68, 25.82, 25.94, 26.05, 26.15, 26.24, 26.31, 26.37, 26.43, 26.48, 26.51, 26.54, 26.57, 26.58, 26.59, 26.6, 26.6, 26.59, 26.58, 26.57, 26.55, 26.53, 26.5, 26.47, 26.44, 26.4, 26.36, 26.32, 26.28, 26.23, 26.18, 26.13, 26.08, 26.02, 25.96, 25.9, 25.84, 25.77, 25.7, 25.63, 25.56, 25.49, 25.41, 25.33, 25.25, 25.17, 25.09, 25.0, 24.92, 24.83, 24.74
    ```
