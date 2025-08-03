import os
import numpy as np
import librosa
import soundfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels
import joblib


# 1. Emotion labels mapping

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    
}

# 2. Feature Extraction Function

def extract_feature(file_name):
    try:
        with soundfile.SoundFile(file_name) as sound:
            X = sound.read(dtype="float32")
            sample_rate = sound.samplerate

            # Pad if audio is shorter than 2048 samples
            if len(X) < 2048:
                pad_width = 2048 - len(X)
                X = np.pad(X, (0, pad_width), mode='constant')

            # Extract 40 MFCCs
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)

            # Convert to 40-length vector
            mfccs_scaled = np.mean(mfccs.T, axis=0)

            # Force shape (40,)
            return np.array(mfccs_scaled, dtype=np.float32).reshape(40,)
    except Exception as e:
        print(f" Skipping {file_name}: {e}")
        return None


# 3. Load Dataset

data_dir = r"C:\Users\mohit\Documents\Emotion_detection\data\Audio_Song_Actors_01-24"

X, y, file_paths = [], [], []

for actor_folder in os.listdir(data_dir):
    actor_path = os.path.join(data_dir, actor_folder)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_path, file)
                mfccs = extract_feature(file_path)
                if mfccs is not None and mfccs.shape == (40,):
                    code = file[6:8]
                    if code in emotion_map:
                        X.append(mfccs)
                        y.append(emotion_map[code])
                        file_paths.append(file_path)

print(" Total valid samples collected:", len(X))

# Convert to arrays safely
X = np.array(X)
y = np.array(y)


# 4. Encode Labels

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 5. Train/Test Split

X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y_encoded, file_paths, test_size=0.2, random_state=42
)


# 6. Train Model

model = RandomForestClassifier()
model.fit(X_train, y_train)


# 7. Evaluate

y_pred = model.predict(X_test)
print("\n Accuracy:", accuracy_score(y_test, y_pred))

labels = unique_labels(y_test, y_pred)
print("\n Classification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=encoder.inverse_transform(labels)
))

# Create Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Plot Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.inverse_transform(labels),
            yticklabels=encoder.inverse_transform(labels))

plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")
plt.title("Confusion Matrix for Speech Emotion Recognition")
plt.show()


# 8. Save Model

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/emotion_model.pkl")
print("\n Model saved in models/emotion_model.pkl")


# 9. Predict for All Audio Files

print("\n Predictions for ALL audio files:\n")
for i, file_path in enumerate(file_paths):
    mfccs = extract_feature(file_path).reshape(1, -1)
    predicted = model.predict(mfccs)
    print(f"{i+1}. {os.path.basename(file_path)} -> {encoder.inverse_transform(predicted)[0]}")


#  Real-Time Emotion Detection

import sounddevice as sd
import tempfile
import wavio
import soundfile as sf
import librosa

def record_and_predict(duration=4, fs=44100):
    print("\n Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print(" Recording finished!")

    # Silence detection
    if np.max(np.abs(recording)) < 0.01:  
        print(" No voice detected. Please try again with clear speech.")
        return

    y_trimmed, _ = librosa.effects.trim(recording.flatten())
    if len(y_trimmed) < 5000:
        print(" Too much silence detected. Try again.")
        return

    temp_file = tempfile.mktemp(".wav")
    sf.write(temp_file, recording, fs)

    mfccs = extract_feature(temp_file).reshape(1, -1)
    predicted = model.predict(mfccs)
    print(" Predicted Emotion:", encoder.inverse_transform(predicted)[0])


#  Keep asking until user exits

while True:
    record_and_predict()
    choice = input("\nDo you want to try again? (y/n): ").lower()
    if choice != 'y':
        print(" Exiting real-time detection.")
        break

