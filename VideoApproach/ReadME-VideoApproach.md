# Football Goal Probability Prediction
## Overview
This project uses deep learning to predict the probability of a goal occurring in football video clips. The solution leverages:
- **Transfer Learning**: Uses MobileNetV2 as a feature extractor
- **Temporal Modeling**: LSTM network to capture sequence patterns in video frames
- **Data Augmentation**: Various techniques to improve model robustness

Performance analysis shows potential optimization opportunities, with model inference taking significantly longer than frame extraction.
## Project Structure
- **`goal_probability_tensor.py`**: Main training script with data loading, model definition, training loop, and visualization
- **`model_evaluation.py`**: Evaluates trained model on test data with classification reports and ROC curves
- **`model_testing.py`**: Inference script for single videos with timing breakdown visualization
- **`updated_file.csv`**: Dataset metadata with video paths and labels
- **`data/`**: Contains video clips organized in and directories `goal/``no_goal/`

## Dependencies
``` bash
pip install tensorflow opencv-python numpy pandas scikit-learn matplotlib
```
## Usage
### 1. Training the Model
Run the training script:
``` bash
python goal_probability_tensor.py
```
This will:
- Load data from `updated_file.csv`
- Train the model
- Save the best model weights to `best_model.h5`
- Generate training/validation plots in `training_metrics.png`

### 2. Evaluating the Model
Edit to set proper paths for test data, then run: `model_evaluation.py`
``` bash
python model_evaluation.py
```
This will:
- Load the trained model from `best_model.h5`
- Generate a classification report
- Calculate ROC AUC score
- Save ROC curve to `roc_curve.png`

### 3. Running Inference
``` bash
python model_testing.py
```
This will:
- Load model from `best_model.h5`
- Perform inference on a sample video
- Calculate time for each processing step
- Generate timing visualizations in `inference_breakdown.png`
- Output the goal probability

## Future Improvements
- **Data Augmentation**: Implement more advanced techniques including synthetic data generation
- **Model Optimization**: Refine architecture and hyperparameters to improve performance and reduce inference time
- **Dataset Expansion**: Collect larger and more diverse dataset
- **Hardware Optimization**: Optimize for specific platforms (GPUs, TPUs)
- : Adapt system for live video analysis **Real-time Processing**
- **Robust Frame Handling**: Improve handling of videos with missing or zero-pixel frames
