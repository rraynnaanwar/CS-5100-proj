# CS5100 Group Project: Soccer Goal Estimation

## Overview
This project estimates the probability of a goal occurring in soccer scenarios. We've explored two distinct approaches:
1. **Single Frame Approach**: Uses a single frame with annotated features along with other data values to predict goal probability
2. **Video Approach**: Analyzes video sequences to determine goal probabilities using deep learning

## Table of Contents
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)

## Usage
### Datasets

- Single Frame Approach: [Drive link](https://drive.google.com/drive/folders/1FUme8Zo5mADquD9FyQXIwVBua6OfmaVL?usp=sharing)
- Video Approach: [Drive link](https://drive.google.com/drive/folders/1xgeI2YpRv0386j57nz_veDZkCk_eTg2I?usp=sharing)

### Single Frame Approach
1. Prepare pitch annotations in JSON format
2. Run the coordinate transformation:
   ```
   cd SingleFrameApproach
   python main.py
   ```
3. Use the Jupyter notebook for xG model training:
   ```
   jupyter notebook xGTraining.ipynb
   ```

### Video Approach
1. Organize video data in the appropriate folders (goal/no_goal/test)
2. Train the goal prediction model:
   ```
   cd VideoApproach
   python goal_probability_tensor.py
   ```
3. Evaluate model performance:
   ```
   python model_evaluation.py
   ```
4. Test inference speed and visualize component breakdown:
   ```
   python model_testing.py
   ```

## Features
- **Coordinate Transformation**: Convert image coordinates to standard pitch coordinates using homography
- **Expected Goals (xG) Calculation**: Calculate xG based on positional data and other features
- **Deep Learning Goal Prediction**: MobileNetV2 + LSTM architecture for video sequence analysis
- **Performance Analysis**: Evaluation metrics and timing breakdown for model components

## Dependencies
- Python 3.9+
- TensorFlow 2.x
- OpenCV
- NumPy
- Pandas
- Matplotlib
- mplsoccer (for visualization)
- Ultralytics YOLO (for object detection)
- scikit-learn
- Jupyter notebooks

## Project Structure
The Video Approach requires data to be added in this structure. 
File Structure
VideoApproach/
├── data/
│   ├── goal/         # Contains positive examples (videos with goals)
│   ├── no_goal/      # Contains negative examples (videos without goals)
│   └── test/         # Contains test data for model evaluation