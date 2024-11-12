# Dynamic-Vision

# Gesture Classification Model

This project is a Transformer-based machine learning model to classify hand gestures using time-series data. The model predicts gestures such as `thumb-up`, `thumb-down`, `palm-left`, `palm-right`, `palm-up`, `palm-down`, `come-forward`, and `go-backward` based on input data collected from various features.

## Prerequisites

To run this project, you need the following dependencies:

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn

You can install the dependencies by running:

```sh
pip install torch numpy pandas scikit-learn
```

## Dataset

The dataset should be in CSV format containing the following columns:

- `rng__no_of_targets`
- `rng__zone_id`
- `peak_rate_kcps_per_spad`
- `range_sigma_mm`
- `median_range_mm`
- `min_range_delta_mm`
- `max_range_delta_mm`
- `target_reflectance_est_pc`
- `target_status`

These features are used to train the model to classify the gestures.

## Code Overview

### Data Preprocessing

1. **Load Dataset**: Load the CSV file containing the dataset.
2. **Feature Selection**: Select relevant features for gesture classification.
3. **Normalization**: Normalize the features using `StandardScaler` to ensure they are on a similar scale.
4. **Label Assignment**: The dataset is assigned labels for different gestures.

### Model Architecture

The model uses a Transformer-based architecture for classification. The architecture consists of:

- **Embedding Layer**: Projects input features to a higher-dimensional space.
- **Transformer Encoder**: Captures relationships between features using self-attention.
- **Fully Connected Layer**: Outputs predictions for the gesture classes.

### Training

- **Loss Function**: Cross-Entropy Loss is used to train the model.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Training Loop**: The model is trained for 10 epochs with a batch size of 32.

### Evaluation

The model is evaluated on a test set, and the accuracy is printed to the console.

### Prediction

You can use the trained model to predict gestures from a new CSV file. The model outputs the predicted gesture based on the provided data.

## How to Run

1. **Update File Paths**: Replace `/path/to/your/combined_gesture_data.csv` with the path to your dataset. Similarly, update `/path/to/your/new_gesture_data.csv` for new predictions.

2. **Run the Script**: Execute the Python script to train the model and make predictions.

```sh
python gesture_classification.py
```

3. **View Results**: After training, the model's accuracy will be printed. You can also make predictions for new data, and the predicted gesture will be displayed.

## Model Parameters

- **Input Dimension**: The number of input features.
- **Number of Classes**: 8 gesture classes.
- **Transformer Parameters**: The model uses 2 encoder layers, 4 attention heads, and a feedforward dimension of 512.

## Improving Accuracy

To improve the model's accuracy, you can consider the following:

- **Increase Dataset Size**: Collect more data for each gesture to improve generalization.
- **Hyperparameter Tuning**: Adjust the learning rate, batch size, and number of epochs.
- **Regularization**: Increase dropout or use L2 regularization to reduce overfitting.
- **Cross-Validation**: Implement k-fold cross-validation to better assess model performance.



