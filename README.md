# Weather Balloon Trajectory Prediction Model

This repository contains a Python-based project designed to predict the trajectories of weather balloons using historical position data and weather conditions from WindBorne and OpenMeteo. The project is structured to fetch balloon position data, interpolate missing data points, and generate training data for an LSTM model. The model is then trained to predict future balloon positions based on past positions and weather conditions.

## Steps:

1. **Data Fetching and Preprocessing**:
   - Fetches balloon position data from WindBorne's live constellation API.
   - Interpolates missing data points using linear interpolation.
   - Converts position data into a structured format suitable for machine learning.

2. **Weather Data Integration**:
   - Integrates weather data (wind speed and direction) from the OpenMeteo API.
   - Estimates atmospheric pressure based on elevation to fetch wind speed and direction data.

3. **Model Training**:
   - Generates input and output vectors for training a machine learning model.
   - Input vectors include balloon positions (x, y, z) and weather data (wind speed, wind direction).
   - Output vectors represent future balloon positions.
   - Uses a neural network model implemented in PyTorch for trajectory prediction.

4. **Model Evaluation**:
   - Evaluate the model's performance using test data.
   - Generates graphs of predicted vs. actual balloon trajectories.

## How to Run the Program

1. **First Run**:
   - Execute the `main.py` file to start the program:
   - This will:
     - Fetch balloon location data and weather data from the APIs.
     - Train the model using the fetched data.
   - **Note**: This process may take some time due to:
     - API throttling by OpenMeteo when querying large amounts of data.
     - Model training time, which depends on your computer's performance.

2. **Subsequent Runs**:
   - After the first run, the location and weather data are saved into two numpy files (`input.npy` and `output.npy`) to avoid re-fetching data.
   - To use these saved files:
     - Comment out everything **except** the `load numpy files` section in the `gen_vectors()` method in `generate_data.py`.
     - Since interpolation is no longer needed, comment out:
       - The constructor of the `GenerateData` class in `generate_data.py`.
       - The line `data.interpolate_missing()` in the `load_and_preprocess_data()` function in `main.py`.

3. **Using a Pre-Trained Model**:
   - During the first run, the trained model is saved as `model.pth`.
   - To use this pre-trained model instead of retraining:
     - Comment out the `train_model()` line in `main()`.

## Caveats

- **Training Data Limitations**:
  - The model is trained on historical data from only **12 of the past 24 hours** (with some missing data points).
  - Weather data resolution is low due to API limitations (updates only when the balloon's position changes by `+= 1` in longitude/latitude).
  
- **Prediction Accuracy**:
  - Predictions may not be entirely accurate due to the above limitations however, the first few predicted hours are somewhat usable.

## Results

The program generates visualizations comparing predicted and actual balloon trajectories. Below are example images of the results: ![Figure_1](https://github.com/user-attachments/assets/8eed28b8-f199-4377-844c-15473a805d0c)
