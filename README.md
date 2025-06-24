# Speech Emotion Classification

## Project Overview

A deep learning system that classifies human emotions from speech audio using Convolutional Neural Networks (CNN). The model can identify 8 different emotions from voice recordings with high accuracy.

**Supported Emotions:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

---

## Model Performance

### Classification Matrix

![Classification Matrix](./images/matrix.png)

### Loss Function Graph

![Loss Graph](./images/loss_graph.png)

---

## Approach

### Feature Extraction

- **MFCC (Mel-Frequency Cepstral Coefficients)**: Captures spectral envelope characteristics
- **Chroma Features**: Represents pitch class information
- **Spectral Features**: Rolloff, centroid, bandwidth for frequency analysis
- **Zero Crossing Rate**: Signal temporal characteristics
- **RMS Energy**: Audio signal power analysis

### CNN Architecture

1. **Input Layer**: Audio features with shape (features, time_steps, 1)
2. **Convolutional Blocks**: 3 blocks with 32, 64, 128 filters respectively
3. **Pooling & Normalization**: Max pooling and batch normalization
4. **Dense Layers**: Fully connected layers with dropout regularization
5. **Output Layer**: Softmax activation for 8-class classification

---

## Project Setup

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd emotion-classification-speech

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Start Streamlit web app
streamlit run app.py

# Test the model
python test_model.py
```

### Project Structure

```
├── app.py                    # Streamlit web application
├── emotion-classification.ipynb  # Training notebook
├── test_model.py            # Model testing script
├── final_emotion_model.h5   # Trained model
├── label_encoder.pkl        # Label encoder
└── requirements.txt         # Dependencies
```

---

## Usage

1. **Web Interface**: Upload audio files or record voice directly
2. **Batch Testing**: Process multiple audio files using test script
3. **Real-time Analysis**: Get instant emotion predictions with confidence scores
