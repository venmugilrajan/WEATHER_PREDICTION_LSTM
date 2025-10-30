import gradio as gr
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
tf.get_logger().setLevel('ERROR')

# --- 1. Load Artifacts (Model and Scaler) ---
# This code runs ONCE when the app starts
try:
    print("Loading pre-trained model...")
    model = tf.keras.models.load_model('lstm_weather_model.keras')
    print("Loading data scaler...")
    scaler = joblib.load('weather_scaler.joblib')
except FileNotFoundError:
    print("---" * 20)
    print("ERROR: Model or scaler file not found.")
    print("Please run 'python train.py' or your notebook first to create the model files.")
    print("---" * 20)
    exit()

print("Gradio app is ready.")

# --- 2. Define the Prediction Function ---
def predict_from_manual_input(input_text):
    """
    Takes a string of 60 comma-separated numbers,
    scales them, and predicts the 61st value.
    """
    # --- Step 1: Parse the user's input text ---
    try:
        # Split the string by commas
        str_values = input_text.split(',')
        # Convert to numbers
        real_values = [float(val.strip()) for val in str_values]
    except Exception as e:
        return f"Error: Could not read your numbers. Make sure they are comma-separated. Details: {e}"
        
    # --- Step 2: Check if we have 60 values ---
    if len(real_values) != 60:
        return f"Error: You entered {len(real_values)} values. The model requires exactly 60."
        
    # --- Step 3: Scale and Reshape the data ---
    try:
        # Convert list to numpy array with the shape (60, 1)
        input_array = np.array(real_values).reshape(-1, 1)
        
        # Scale the data using the loaded scaler
        scaled_input = scaler.transform(input_array)
        
        # Reshape for the model: [samples, time_steps, features] -> (1, 60, 1)
        model_input = scaled_input.reshape(1, 60, 1)
        
    except Exception as e:
        return f"Error during scaling. Did you train the scaler? Details: {e}"

    # --- Step 4: Make the Prediction ---
    try:
        # Make prediction (output is scaled)
        scaled_prediction = model.predict(model_input)
        
        # Inverse transform to get the real temperature value
        real_prediction = scaler.inverse_transform(scaled_prediction)
        
        return f"Predicted Temperature: {real_prediction[0][0]:.2f} °C"
        
    except Exception as e:
        return f"Error during prediction. Model may be corrupt. Details: {e}"

# --- 3. Create the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🌡️ Manual Weather Forecaster (LSTM)
        This model requires **exactly 60** past temperature values to predict the next one.
        """
    )
    
    # The user input box
    temp_input_box = gr.Textbox(
        lines=5,
        label="Enter 60 Comma-Separated Temperatures",
        placeholder="Example: 2.1, 2.2, 2.3, 2.1, 2.0, ..."
    )
    
    # Prediction button and output
    predict_button = gr.Button("Forecast Next Temperature")
    output_textbox = gr.Textbox(
        label="Predicted Temperature",
        interactive=False
    )
    
    # --- Click Action ---
    predict_button.click(
        fn=predict_from_manual_input,
        inputs=[temp_input_box],
        outputs=[output_textbox]
    )

# --- 4. Launch the App ---
if __name__ == "__main__":
    app.launch()