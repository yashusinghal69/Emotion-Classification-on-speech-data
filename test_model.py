import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from tqdm import tqdm
import glob
import seaborn as sns

# Set up argument parser
parser = argparse.ArgumentParser(description='Test emotion classification model on audio files')
parser.add_argument('--model_path', type=str, default='best_emotion_model.h5', help='Path to the model file')
parser.add_argument('--encoder_path', type=str, default='label_encoder.pkl', help='Path to the label encoder file')
parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test audio files')
parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save test results')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Enhanced feature extraction function
def extract_enhanced_features(file_path, max_pad_len=174):
    try:
        # Load audio with higher sample rate and better parameters
        audio, sample_rate = librosa.load(file_path, sr=22050, res_type='kaiser_fast')
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Extract multiple features with error handling
        features_list = []
        
        # 1. MFCC features (40 coefficients)
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
            try:
                chroma = librosa.feature.chromagram(y=audio, sr=sample_rate)
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
            features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Pad or truncate to fixed length
        pad_width = max_pad_len - features.shape[1]
        if pad_width > 0:
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :max_pad_len]
            
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def parse_filename(filepath):
    """Parse RAVDESS filename to extract emotion label"""
    try:
        filename = os.path.basename(filepath)
        parts = filename.split('.')[0].split('-')
        
        # RAVDESS filename format: 03-01-01-01-01-01-01.wav
        # where the 3rd element is emotion code
        emotion_code = parts[2]
        
        emotion_map = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        return emotion_map.get(emotion_code, 'unknown')
    except:
        return 'unknown'

def main():
    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path)
        le = joblib.load(args.encoder_path)
    except Exception as e:
        print(f"Error loading model or encoder: {e}")
        return
    
    # Get list of audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.ogg']:
        audio_files.extend(glob.glob(os.path.join(args.test_dir, ext)))
    
    if not audio_files:
        print(f"No audio files found in {args.test_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files. Processing...")
    
    # Extract features and get true labels
    X_test = []
    y_true = []
    filenames = []
    
    for file in tqdm(audio_files):
        features = extract_enhanced_features(file)
        if features is not None:
            X_test.append(features)
            true_label = parse_filename(file)
            y_true.append(true_label)
            filenames.append(os.path.basename(file))
    
    if not X_test:
        print("No valid features could be extracted from the audio files.")
        return
    
    # Convert to numpy arrays
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    # Make predictions
    print("Making predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    y_pred = le.inverse_transform(y_pred_indices)
    
    # Filter out unknown labels
    valid_indices = [i for i, label in enumerate(y_true) if label != 'unknown']
    if not valid_indices:
        print("No files with valid emotion labels found.")
        return
    
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    filenames_valid = [filenames[i] for i in valid_indices]
    
    # Check if we have valid labels in le.classes_
    valid_labels = [label for label in y_true_valid if label in le.classes_]
    if not valid_labels:
        print("None of the true labels match the model's known classes.")
        print(f"Model classes: {le.classes_}")
        print(f"Found labels: {set(y_true_valid)}")
        return
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Filename': filenames,
        'True_Emotion': y_true,
        'Predicted_Emotion': y_pred,
        'Confidence': np.max(y_pred_probs, axis=1)
    })
    
    # Save results to CSV
    results_csv = os.path.join(args.output_dir, 'test_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")
    
    # Create confusion matrix for valid labels
    try:
        # Get labels that are both in true labels and model classes
        common_labels = sorted(list(set(y_true_valid) & set(le.classes_)))
        
        # Filter data to only include common labels
        valid_indices = [i for i, label in enumerate(y_true_valid) if label in common_labels]
        y_true_common = [y_true_valid[i] for i in valid_indices]
        y_pred_common = [y_pred_valid[i] for i in valid_indices]
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_common, y_pred_common, labels=common_labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=common_labels, yticklabels=common_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Emotion')
        plt.ylabel('True Emotion')
        plt.tight_layout()
        
        # Save confusion matrix
        plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
        print(f"Confusion matrix saved to {args.output_dir}/confusion_matrix.png")
        
        # Classification report
        report = classification_report(y_true_common, y_pred_common, labels=common_labels)
        with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        print("\nClassification Report:")
        print(report)
        
        # Calculate overall accuracy
        accuracy = np.mean(np.array(y_true_common) == np.array(y_pred_common))
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error generating metrics: {e}")
    
    # Create a bar chart of emotion distribution
    plt.figure(figsize=(12, 6))
    emotion_counts = pd.Series(y_true_valid).value_counts()
    emotion_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Emotions in Test Set')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'emotion_distribution.png'))

if __name__ == "__main__":
    main()
