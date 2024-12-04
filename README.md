# Urban Heat Island (UHI) Detection Using LiDAR and Thermal Data
This project aims to identify Urban Heat Island (UHI) effects through predictive modeling, leveraging the NYC4Urban multimodal dataset. The study utilizes LiDAR-derived features and thermal layers to predict temperature distributions, focusing on mean and standard deviation values.

# Project Overview
Urban Heat Islands are regions with significantly higher temperatures than their surroundings, contributing to challenges such as:

Increased energy consumption
Elevated pollution levels
Heat-related illnesses
By identifying UHI regions, urban planners can implement mitigation strategies such as:

Cooling techniques like green spaces
Public health preparedness
Heat-resilient infrastructure

# Dataset
The project uses the NYC4Urban dataset, comprising:

LiDAR Layers (9-20): Includes return count, elevation, and reflectance statistics.
Thermal Layers (23-25): Temperature measurements from Landsat-8. Layers 26 and 27 were excluded due to data quality issues.

# Preprocessing
Missing data in LiDAR layers was handled using cv2.inpaint for spatial continuity.
Resolution mismatch between LiDAR (0.3m) and Thermal (100m) layers required reformulation of the prediction task.

# Approach
## Model Architecture
A CNN-based regression model predicts the mean and standard deviation of temperature from LiDAR layers. Key components include:

Convolutional Layers: Extract spatial features with 16–64 filters.
Batch Normalization: Standardizes inputs and accelerates convergence.
Global Average Pooling: Reduces dimensionality before dense layers.
Dense Layers: Two fully connected layers predict the final temperature statistics.

## Training Details
Loss Function: Mean Squared Error (MSE)
Optimizer: Adam
Batch Size: 16
Validation Split: 20%
Early Stopping: Stops training after 10 epochs of no improvement in validation loss.
Checkpoints: Saves the best-performing model.

# Results
Correlation Analysis
LiDAR layers with the highest correlation to temperature:

Elevation (Mean)
Elevation (Max)
Elevation (Min)
These layers capture information strongly related to surface temperature patterns.

# Model Performance
The model successfully predicted:

Mean Temperature: Example - Original: 29.67°C, Predicted: 28.63°C
Standard Deviation: Example - Original: 1.21°C, Predicted: 1.31°C

# Visualizations
Example Outputs
LiDAR Elevation Mean: Input visualization of LiDAR data.
Thermal Maps: Corresponding thermal layers with derived mean and standard deviation values.
Correlation Plots: Relationship between LiDAR features and thermal statistics.
Model Prediction Plots: Comparison of predicted vs. original mean and standard deviation values.

# Challenges
Missing Data: Resolved using interpolation.
Resolution Mismatch: Refocused prediction task from pixelwise temperature mapping to statistical pattern prediction.
Compute Limitations: Restricted training dataset size to avoid crashes.

# Future Steps
Augmented Training Dataset: Incorporate more samples to improve generalization.
Increased Model Complexity: Experiment with deeper architectures or ensemble models.
Hyperparameter Tuning: Optimize learning rate, batch size, and other parameters.
Real-World Applications: Integrate the model with urban planning frameworks for actionable insights.

