import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from tensorflow.keras.callbacks import EarlyStopping

# Dataset downloaded from Kaggle: phishing_site_urls.csv
# Columns are URL and Label
data = pd.read_csv("phishing_site_urls.csv")
urls = data['URL']
labels = data['Label']
labels = labels.apply(lambda x: 1 if x.strip().lower() == 'bad' else 0) # Numeric labels

# Preprocessing
tokenizer = Tokenizer(char_level=True) 
tokenizer.fit_on_texts(urls)
sequences = tokenizer.texts_to_sequences(urls)
max_length = 150
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

# Casting labels to floats
y_train = np.array(y_train).astype(float)
y_test = np.array(y_test).astype(float)

# LSTM Setup, used resource in project description for guidance: 
# https://github.com/christianversloot/machine-learning-articles/blob/main/build-an-lstm-model-with-tensorflow-and-keras.md

embedding_dim = 128
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_length),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training setup
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
batch_size = 64
epochs = 10
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate using sklearn metrics
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
model.save("detect_phish.h5")