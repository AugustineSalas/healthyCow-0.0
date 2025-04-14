import os
import pandas as pd
import numpy as np
from math import ceil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small  # or MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    Callback
)
from sklearn.metrics import f1_score

# ------------------------------
# 1. Check GPU Availability
# ------------------------------
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# ------------------------------
# 2. Build a DataFrame from Directory Structure
# ------------------------------
data_dir = 'dataset'
classes = ['infected', 'normal']
data = []

for label in classes:
    folder = os.path.join(data_dir, label)
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append({'filepath': os.path.join(folder, fname), 'label': label})

df = pd.DataFrame(data)

# ------------------------------
# 3. Split the Data
# ------------------------------
df_train_val, df_test = train_test_split(
    df, test_size=0.15, stratify=df['label'], random_state=42
)
df_train, df_val = train_test_split(
    df_train_val, test_size=0.15, stratify=df_train_val['label'], random_state=42
)

print(f"Total images: {len(df)}")
print(f"Training images: {len(df_train)}")
print(f"Validation images: {len(df_val)}")
print(f"Test images: {len(df_test)}")

# ------------------------------
# 4. Create Data Generators
# ------------------------------
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col='filepath',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Generator used by model.fit for validation
validation_generator = val_test_datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col='filepath',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# **Separate** generator for the F1 callback to avoid "using up" the validation data
validation_generator_for_callback = val_test_datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col='filepath',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col='filepath',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# ------------------------------
# 5. Custom F1 Callback
# ------------------------------
class F1ScoreCallback(Callback):
    """
    Uses a separate validation generator so we do not interfere 
    with the validation generator used by 'fit' for val_loss/val_acc.
    """
    def __init__(self, val_gen, val_steps):
        super().__init__()
        self.val_gen = val_gen
        self.val_steps = val_steps

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Reset generator at start of epoch end to ensure we get full data
        self.val_gen.reset()
        true_labels = []
        predictions = []
        
        for _ in range(self.val_steps):
            x_val, y_val = next(self.val_gen)
            y_pred = self.model.predict_on_batch(x_val)
            y_pred_binary = (y_pred > 0.5).astype(int)
            true_labels.extend(y_val)
            predictions.extend(y_pred_binary)

        true_labels = np.array(true_labels)
        predictions = np.array(predictions)
        f1 = f1_score(true_labels, predictions, average='binary')
        logs['val_f1_score'] = f1
        print(f" - val_f1_score: {f1:.4f}")

# ------------------------------
# 6. Build and Compile Model
# ------------------------------
base_model = MobileNetV3Small(weights='imagenet', include_top=False, 
                              input_shape=(img_size[0], img_size[1], 3))
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ------------------------------
# 7. Training
# ------------------------------
train_steps = ceil(len(df_train) / batch_size)
val_steps = ceil(len(df_val) / batch_size)
test_steps = ceil(len(df_test) / batch_size)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-7)
f1_callback = F1ScoreCallback(validation_generator_for_callback, val_steps)

epochs = 30

print("Starting training with frozen base layers...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=[early_stop, checkpoint, reduce_lr, f1_callback]
)

# Evaluate on test data before fine tuning
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# ------------------------------
# 8. Fine Tuning
# ------------------------------
base_model.trainable = True
fine_tune_at = 100  # Freeze all layers before this index
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Starting fine tuning...")
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=[early_stop, checkpoint, reduce_lr, f1_callback]
)

# Final evaluation on test data after fine tuning
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
print(f"After fine tuning - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
