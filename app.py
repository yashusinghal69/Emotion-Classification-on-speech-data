import streamlit as st
import numpy as np
import librosa
import os
import tensorflow as tf
import joblib
import tempfile
import warnings
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Set page configuration
st.set_page_config(
    page_title="Speech Emotion Classifier",
    page_icon="🎭",
    layout="wide"
)

@st.cache_resource
def load_model_and_encoder():
    """Load the model and label encoder with caching"""
    model_path = "final_emotion_model.h5"

    # Load model with custom objects if needed
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load label encoder
    label_encoder = joblib.load("label_encoder.pkl")
    
    return model, label_encoder, None
        

def extract_features(file_path, max_pad_len=174):
    """Extract audio features for prediction - matching training data"""
    audio, sample_rate = librosa.load(file_path, sr=22050, res_type='kaiser_fast')
    
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    features_list = []
    
    # 1. MFCC features 
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=2048, hop_length=512)
        features_list.append(mfcc)
    except:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        features_list.append(mfcc)
    
    # 2. Chroma features 
    try:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, hop_length=512)
        features_list.append(chroma)
    except:
        pass

    # 3. Spectral contrast
    try:
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, hop_length=512)
        features_list.append(spectral_contrast)
    except:
        pass
    
    # 4. Zero crossing rate
    try:
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=512)
        features_list.append(zcr)
    except:
        pass
    
    # 5. Spectral rolloff 
    try:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, hop_length=512)
        features_list.append(spectral_rolloff)
    except:
        pass
    
    # 6. Spectral centroid
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, hop_length=512)
        features_list.append(spectral_centroid)
    except:
        pass
    
    # 7. RMS Energy
    try:
        rms = librosa.feature.rms(y=audio, hop_length=512)
        features_list.append(rms)
    except:
        pass
    
    # 8. Spectral bandwidth
    try:
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, hop_length=512)
        features_list.append(spectral_bandwidth)
    except:
        pass
    
    # 9. Mel-frequency spectral coefficients
    try:
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=13, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features_list.append(mel_spec_db)
    except:
        pass
    
    # Combine all available features
    if len(features_list) > 0:
        min_frames = min([feat.shape[1] for feat in features_list])
        features_list = [feat[:, :min_frames] for feat in features_list]
        features = np.vstack(features_list)
    else:
        # Fallback to just MFCC if other features fail
        features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
    # Pad or truncate to fixed length
    if features.shape[1] < max_pad_len:
        pad_width = max_pad_len - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_pad_len]
    
    #
    if features.shape[0] < 77:
        pad_width = 77 - features.shape[0]
        features = np.pad(features, pad_width=((0, pad_width), (0, 0)), mode='constant')
    elif features.shape[0] > 77:
        features = features[:77, :]
        
    return features, audio, sample_rate
        

def predict_emotion(audio_file, model, label_encoder):
    """Make emotion prediction"""
    features, audio, sample_rate = extract_features(audio_file)
    if features is None:
        return None, None, None
    
    # Reshape for model input: (batch_size, 77, 174, 1)
    features = features.reshape(1, features.shape[0], features.shape[1], 1)
    
    # Make prediction
    with st.spinner("Analyzing emotion..."):
        prediction = model.predict(features, verbose=0)[0]
    
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    
    return predicted_class, prediction, (audio, sample_rate)



def create_emotion_chart(predicted_class, prediction, label_encoder):
    emotions = label_encoder.classes_
    
    # Create bar chart
    emotion_df = pd.DataFrame({
        'Emotion': emotions,
        'Probability': prediction
    }).sort_values('Probability', ascending=True)
    
    fig = px.bar(emotion_df, 
                 x='Probability', 
                 y='Emotion', 
                 orientation='h',
                 title=f'Detected Emotion: {predicted_class.upper()}',
                 color='Probability',
                 color_continuous_scale='viridis')
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def create_waveform_chart(audio, sample_rate):
    time = np.arange(len(audio)) / sample_rate
    
    fig = px.line(x=time, y=audio, 
                  labels={'x': 'Time (s)', 'y': 'Amplitude'},
                  title='Audio Waveform')
    fig.update_layout(height=300)
    
    return fig

def main():
    model, label_encoder, error = load_model_and_encoder()
    
    if error:
        st.error(error)
        return
    
    # App header
    st.title("🎭 Speech Emotion Recognition")
    st.markdown("Upload an audio file to analyze the emotion in speech")
    
    # File uploader
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV, MP3, OGG)", 
        type=['wav', 'mp3', 'ogg'],
        help="For best results, use clear audio with minimal background noise"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_filename = tmp.name
        
        # Analyze button
        if st.button("🔍 Analyze Emotion", type="primary"):
            try:
                # Make prediction
                predicted_class, prediction, audio_data = predict_emotion(
                    temp_filename, model, label_encoder
                )
                
                if predicted_class is not None:
                    # Display results
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    
                    # Main result
                    st.success(f"**Detected Emotion: {predicted_class.upper()}**")
                    
                    # Create two columns for charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Emotion probabilities chart
                        emotion_fig = create_emotion_chart(predicted_class, prediction, label_encoder)
                        st.plotly_chart(emotion_fig, use_container_width=True)
                    
                    with col2:
                        # Audio waveform
                        audio, sample_rate = audio_data
                        waveform_fig = create_waveform_chart(audio, sample_rate)
                        st.plotly_chart(waveform_fig, use_container_width=True)
                    
                    # Show confidence scores
                    st.subheader("Confidence Scores:")
                    emotions = label_encoder.classes_
                    for emotion, prob in zip(emotions, prediction):
                        st.write(f"**{emotion.capitalize()}:** {prob:.3f}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                else:
                    st.error("Failed to analyze audio. Please try a different file.")
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_filename)
                except:
                    pass
                
if __name__ == "__main__":
    main()
