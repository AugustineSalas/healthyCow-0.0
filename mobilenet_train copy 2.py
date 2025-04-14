import os
import pandas as pd
import numpy as np
from math import ceil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau
)
from tensorflow.keras.metrics import AUC
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from sklearn.utils.class_weight import compute_class_weight

# ------------------------------
# 1. Check GPU Availability
# ------------------------------
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# ------------------------------
# 2. Build a DataFrame from Directory Structure
# ------------------------------
data_dir = 'newdataset'
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
# 4. Compute Adjusted Class Weights
# ------------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df_train['label']),
    y=df_train['label']
)
class_weights = {0: class_weights[0] * 1.5, 1: class_weights[1] * 0.75}  # Increase infected class weight
print("Class Weights:", class_weights)

# ------------------------------
# 5. Create Data Generators
# ------------------------------
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.4,
    brightness_range=[0.7, 1.3],
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

validation_generator = val_test_datagen.flow_from_dataframe(
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
# 6. Build and Compile Optimized MobileNetV2 Model
# ------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
base_model.trainable = False  # Freeze base model initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy', AUC(name='auc')])

# ------------------------------
# 7. Training with Frozen Base Layers
# ------------------------------
train_steps = ceil(len(df_train) / batch_size)
val_steps = ceil(len(df_val) / batch_size)
test_steps = ceil(len(df_test) / batch_size)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-7)

epochs = 40

print("Starting training with frozen base layers...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weights
)

test_results = model.evaluate(test_generator, steps=test_steps)
print(f"Test Results: {dict(zip(model.metrics_names, test_results))}")

# ------------------------------
# 8. Fine Tuning
# ------------------------------
base_model.trainable = True
fine_tune_at = 75  # Unfreeze more layers for better learning
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

for layer in base_model.layers[fine_tune_at:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=Adam(learning_rate=5e-6),
              loss='binary_crossentropy',
              metrics=['accuracy', AUC(name='auc')])

print("Starting fine tuning...")
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weights
)

# ------------------------------
# 9. Plot Training and Fine-Tuning Metrics
# ------------------------------
def plot_training_history(history, history_fine):
    def plot_metric(metric, title):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[metric], label='Training ' + metric)
        plt.plot(history.history['val_' + metric], label='Validation ' + metric)
        plt.plot(history_fine.history[metric], label='Fine-Tuned Training ' + metric)
        plt.plot(history_fine.history['val_' + metric], label='Fine-Tuned Validation ' + metric)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

    for metric in ['accuracy', 'loss', 'auc']:
        plot_metric(metric, f'{metric.capitalize()} over Epochs')

# ------------------------------
# 10. Compute F1 Score on Test Set
# ------------------------------
test_generator.reset()
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator, steps=test_steps)
y_pred = (y_pred_prob > 0.5).astype(int)

f1 = f1_score(y_true, y_pred)
print(f"F1 Score on Test Set: {f1:.4f}")



plot_training_history(history, history_fine)


test_results = model.evaluate(test_generator, steps=test_steps)
print(f"Test Results: {dict(zip(model.metrics_names, test_results))}")

