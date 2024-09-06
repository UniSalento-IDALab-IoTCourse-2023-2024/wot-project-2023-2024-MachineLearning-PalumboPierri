import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import os

# Funzione per estrarre le caratteristiche audio
def extract_features(y, sr):
    features = []
    try:
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.append(np.mean(mfccs.T, axis=0))

        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(np.mean(chroma_stft.T, axis=0))

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(spectral_centroid))

        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(np.mean(spectral_bandwidth))

        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), n_bands=6)
        features.append(np.mean(spectral_contrast.T, axis=0))

        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        features.append(np.mean(rolloff))

        # RMS
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))

        # Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        features.append(np.mean(mel_spectrogram.T, axis=0))

        # Concatenate all features into a single array
        features = np.concatenate([np.ravel(f) for f in features])
    except Exception as e:
        print(f"Error computing features: {e}")
        return np.array([])
    return features

# Funzione per l'audio augmentation
def augment_audio(y, sr):
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
    y_stretch = librosa.effects.time_stretch(y, rate=0.8)
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise
    return [y_pitch, y_stretch, y_noise]

# Trova il percorso del file audio all'interno delle sottocartelle
def find_audio_file(base_path, file_name):
    for root, dirs, files in os.walk(base_path):
        if file_name in files:
            return os.path.join(root, file_name)
    return None

# Carica il file di metadati
metadata = pd.read_csv('C:/Users/39346/Desktop/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv')

# Verifica le colonne del CSV
print("Colonne del CSV:", metadata.columns)

# Lista per le caratteristiche e le etichette
X = []
y = []

# Base path della cartella audio
audio_base_path = 'C:/Users/39346/Desktop/UrbanSound8K/UrbanSound8K/audio/'

# Estrazione delle caratteristiche e delle etichette
for index, row in metadata.iterrows():
    file_name = row["slice_file_name"]
    file_path = find_audio_file(audio_base_path, file_name)
    if file_path:
        print(f"Elaborazione del file: {file_path}")
        # Carica il file audio originale
        y_raw, sr = librosa.load(file_path, sr=None)
        # Estrai le caratteristiche dal file audio originale
        features = extract_features(y_raw, sr)
        if features.size > 0 and not np.isnan(features).any() and not np.isinf(features).any():
            X.append(features)
            y.append(row['class'])
        else:
            print(f"Caratteristiche vuote o non valide per il file: {file_path}")

        # Applica l'audio augmentation
        augmented_signals = augment_audio(y_raw, sr)

        # Estrai le caratteristiche dagli audio augmentati senza salvarli su disco
        for i, y_aug in enumerate(augmented_signals):
            features_aug = extract_features(y_aug, sr)
            if features_aug.size > 0 and not np.isnan(features_aug).any() and not np.isinf(features_aug).any():
                X.append(features_aug)
                y.append(row['class'])
            else:
                print(f"Caratteristiche vuote o non valide per il file augmentato {i}")
    else:
        print(f"File non trovato: {file_name}")

# Stampa la dimensione di X e y per il debug
print(f"Dimensione di X: {len(X)}")
print(f"Dimensione di y: {len(y)}")

# Verifica se X e y contengono dati
if len(X) == 0 or len(y) == 0:
    raise ValueError("Nessun dato disponibile per l'addestramento")

X = np.array(X)
y = np.array(y)

# Codifica delle etichette
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = np.eye(len(np.unique(y)))[y]  # One-hot encoding

# Suddivisione in set di addestramento e test
if len(X) > 0 and len(y) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    raise ValueError("Impossibile suddividere i dati in set di addestramento e test.")

# Costruzione del modello
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training del modello
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Valutazione del modello
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# Salva il modello
model.save('audio_classification_model.h5')
print("Modello salvato con successo.")

# Percorso del nuovo file audio
new_file_path = 'C:/Users/39346/Downloads/drill.wav'  # Sostituisci con il nome del tuo file

# Estrai le caratteristiche dal nuovo file audio
new_y, new_sr = librosa.load(new_file_path, sr=None)
new_features = extract_features(new_y, new_sr)
if new_features.size == 0:
    raise ValueError("Errore nell'estrazione delle caratteristiche dal nuovo file audio.")

# Converti le caratteristiche in un tensor di TensorFlow
new_features_tensor = np.expand_dims(new_features, axis=0)  # Aggiungi una dimensione per il batch

# Fai una previsione
prediction = model.predict(new_features_tensor)

# Ottieni la categoria con la probabilità più alta
predicted_class = np.argmax(prediction, axis=1)

# Decodifica la previsione per ottenere la classe effettiva
predicted_label = label_encoder.inverse_transform(predicted_class)

print(f"Il suono è stato classificato come: {predicted_label[0]}")
