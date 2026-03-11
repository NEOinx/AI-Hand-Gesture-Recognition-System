import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode the string labels to numbers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# Build the neural network
model = Sequential([
    Dense(128, input_shape=(42,), activation='relu'), # Input layer for 42 landmark values
    Dense(64, activation='relu'),
    Dense(len(np.unique(labels_encoded)), activation='softmax') # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Save the trained model
model.save('gesture_model.h5')

# Save the label encoder
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model and label encoder saved successfully.")