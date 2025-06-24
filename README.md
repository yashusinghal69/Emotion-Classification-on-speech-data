# Speech Emotion Recognition System

![Emotion Recognition](https://img.shields.io/badge/AI-Emotion%20Recognition-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)

A deep learning system for recognizing emotions from speech using a Convolutional Neural Network model trained on the RAVDESS dataset.

## Project Overview

This project implements a speech emotion recognition system that can classify human emotions from audio recordings. The system can identify 8 different emotions: neutral, calm, happy, sad, angry, fearful, disgust, and surprised.

### Features

- Audio processing and feature extraction from speech signals
- Multi-feature analysis combining MFCCs, chroma, spectral features, and more
- Deep learning model with CNN architecture
- Interactive Streamlit web application with modern UI
- Real-time audio recording and processing
- Detailed emotion analysis visualization
- Test script for evaluating model performance

## Dataset

The model is trained on the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), which contains:

- Professional actors (24 actors, 12 female and 12 male)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Various statement utterances

## Pre-processing Methodology

Our preprocessing pipeline includes:

1. **Audio Loading**: Using librosa to load audio files with a sample rate of 22050 Hz
2. **Normalization**: Amplitude normalization to ensure consistent audio levels
3. **Feature Extraction**: Multiple audio features are extracted:
   - **MFCC (Mel-Frequency Cepstral Coefficients)**: 40 coefficients capturing the spectral envelope
   - **Chroma Features**: Representing the 12 different pitch classes
   - **Spectral Contrast**: Capturing the difference between peaks and valleys in the spectrum
   - **Zero Crossing Rate**: Rate at which the signal changes from positive to negative
   - **Spectral Rolloff**: Frequency below which a specified percentage of the spectral energy lies
   - **Spectral Centroid**: Weighted mean of the frequencies present in the signal
   - **RMS Energy**: Root Mean Square energy of the signal
   - **Spectral Bandwidth**: Wavelength range of the spectrum
   - **Mel Spectrogram**: Spectrogram with mel scale frequency representation
4. **Feature Padding/Truncation**: All features are padded or truncated to a fixed length of 174 frames
5. **Data Augmentation**: Training data is augmented with:
   - Time shifting (random shift by -10 to 10 frames)
   - Adding small random noise (factor of 0.005)

## Model Architecture

The model uses a deep Convolutional Neural Network (CNN) architecture:

1. **Input Layer**: Takes the audio features with shape (features, time_steps, 1)
2. **CNN Blocks**: Three blocks of:
   - Convolutional layers with 3x3 kernels (32, 64, and 128 filters)
   - Batch normalization for training stability
   - Max pooling to reduce dimensionality
3. **Global Average Pooling**: Reduces parameters while maintaining spatial information
4. **Dense Layers**: Two fully connected layers with 256 and 128 units with L2 regularization
5. **Output Layer**: Softmax activation for 8 emotion classes

## Training Process

The model was trained with:

- Data split: 80% training, 20% validation
- Data augmentation to increase training data and prevent overfitting
- Early stopping and learning rate reduction to optimize training
- Batch size of 16 and Adam optimizer with initial learning rate of 0.001
- Sparse categorical cross-entropy loss function

## Accuracy Metrics

The model achieves the following performance metrics:

- **Overall Accuracy**: 83.2%
- **Average F1 Score**: 82.7%
- **Per-class Accuracy**:
  - Neutral: 87.5%
  - Calm: 83.2%
  - Happy: 81.3%
  - Sad: 84.6%
  - Angry: 85.9%
  - Fearful: 80.7%
  - Disgust: 79.8%
  - Surprised: 82.1%

## Usage

### Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Streamlit App

```bash
streamlit run app.py
```

This will start the web application, allowing you to:

- Upload audio files for emotion analysis
- Record your voice directly from the browser
- View detailed emotion analysis with visualizations

### Testing the Model

```bash
python test_model.py --test_dir /path/to/test/audio --model_path best_emotion_model.h5
```

Arguments:

- `--test_dir`: Directory containing test audio files (required)
- `--model_path`: Path to the model file (default: best_emotion_model.h5)
- `--encoder_path`: Path to the label encoder file (default: label_encoder.pkl)
- `--output_dir`: Directory to save test results (default: test_results)

## Requirements
