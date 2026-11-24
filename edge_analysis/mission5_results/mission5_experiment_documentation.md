
# EDGE AI TEAM 13 - MISSION 5
## Original and Cleaned Dataset Model Experiments

### 1. Dataset information

Original dataset records: 2000

Cleaned dataset records: 1976

Rows removed: 24

Common validation records: 395

### 2. Cleaning rules

- Remove zero battery voltage.
- Remove temperature less than or equal to zero.
- Remove zero WiFi RSSI.
- Remove negative rings per day.
- Remove negative device age.
- Remove negative remaining battery life.
- Preserve zero rings per day.
- Preserve zero device age.
- Original dataset is not overwritten.

### 3. Experiment methodology

Random seed: 42

Training fraction: 80 percent of original row indices.

Validation population: 395 common valid records.

Features: battery_v, temp_c, rings_per_day, wifi_rssi_dbm, age_months

Target: days_left

Scaling: StandardScaler fitted on training data only.

Optimizer: Adam.

Loss: Mean Squared Error.

Learning rate: 0.01.

Weight decay: 0.0001.

Maximum epochs: 1500.

Early stopping patience: 150.

### 4. Architectures tested

One hidden layer: 4, 8, 16, 32, 64 neurons.

Two hidden layers: 4-4, 8-8, 16-16, 32-32, 64-64 neurons.

Additional models: Mean Baseline and Linear Regression.

### 5. Best observed original model

Model: NN_2L_64_64

Architecture: 5 -> 64 -> 64 -> 1

Training MAE: 4.0100

Validation MAE: 5.0811

Validation RMSE: 7.7878

Validation R2: 0.9878

### 6. Best observed cleaned model

Model: NN_2L_64_64

Architecture: 5 -> 64 -> 64 -> 1

Training MAE: 3.6015

Validation MAE: 5.1325

Validation RMSE: 7.6512

Validation R2: 0.9882

### 7. Output files

mission5_train_validation_complete.csv

mission5_before_after_mae.csv

mission5_overfitting_gap_analysis.csv

mission5_depth_comparison.csv

mission5_train_validation_mae.png

mission5_original_cleaned_mae.png

### 8. Limitations

- Results use one random split.
- Validation data is used for early stopping and model selection.
- Validation results are not independent test-set results.
- Original and cleaned training populations differ.
- Cleaning may affect different model architectures differently.
- Lower validation MAE does not independently establish better real-world generalization.
- Larger networks require more parameters and computational resources.
