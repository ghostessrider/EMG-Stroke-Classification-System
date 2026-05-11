# EMG Stroke Classification System

This project is a pipeline for classifying muscle activity as either "Post-Stroke" or "Healthy" using Electromyography (EMG) signals. It was developed to handle the full process of medical data analysis: from cleaning raw sensor data to deploying a model that can make near-instant predictions.

The system is built around the dataset from Lucchetti et al. (2025) and was completed as part of the EEL208 course at IIT Bhilai.

## Implementation Details

### 1. Signal Processing
Raw EMG data is often messy and full of noise. This system implements a multi-stage filtering pipeline:
* **Bandpass Filtering (20–450 Hz):** To remove motion artifacts and high-frequency noise.
* **Notch Filtering (50 Hz):** Specifically designed to remove power-line interference.
* **Rectification & Envelope Extraction:** Full-wave rectification followed by a 6 Hz low-pass filter to reveal the underlying muscle activation patterns.

### 2. Feature Extraction
I extracted 12 different features per channel across both time and frequency domains to ensure the classifier has a robust view of the signal:
* **Time Domain:** RMS, MAV, Waveform Length, Zero Crossings, SSC, and Variance.
* **Frequency Domain:** Mean Frequency (MNF), Median Frequency (MDF), and Peak Frequency.
* **Activation:** Envelope Peak and Mean.

### 3. Machine Learning Architecture
Instead of relying on a single model, the system uses an ensemble approach with a majority-vote of:
* Support Vector Machine (SVM) with an RBF kernel.
* Random Forest Classifier.
* Logistic Regression.

The project is structured with a "Production" mindset, featuring two distinct modes:
* **TRAIN:** Processes the raw dataset, extracts features, and saves the trained model weights.
* **PREDICT:** Loads the pre-trained weights for fast classification of new subject data without needing to re-process the entire dataset.

## Project Structure

* `convert.py`: Converts raw laboratory `.mat` files into manageable CSV formats.
* `emg_project.py`: The core engine for filtering, feature extraction, and classification.
* `/saved_model/`: Stores the trained `.joblib` files (SVM, RF, Scaler).
* `/outputs/`: Visual reports including PCA plots, signal comparisons, and confusion matrices.

## How to Run

### Setup
```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib
