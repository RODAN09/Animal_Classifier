import tensorflow as tf
from tensorflow.keras import layers, models

# ✅ Dataset path
data_dir = "dataset"
img_size = (224, 224)
batch_size = 32

# ✅ Load dataset (first)
train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)
val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# ✅ Get class names before mapping
class_names = train_ds_raw.class_names
print("✅ Class Names:", class_names)

# ✅ Ensure all images are converted to RGB
def convert_to_rgb(image, label):
    image = tf.image.grayscale_to_rgb(image) if image.shape[-1] == 1 else image
    return image, label

# ✅ Apply the RGB conversion
train_ds = train_ds_raw.map(convert_to_rgb)
val_ds = val_ds_raw.map(convert_to_rgb)

# ✅ Optimize dataset pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ✅ Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ✅ Pretrained base (EfficientNetB0)
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=img_size + (3,)
)
base_model.trainable = False

# ✅ Build Model
inputs = layers.Input(shape=img_size + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

# ✅ Compile
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ✅ Train (for quick test: use 5–10 epochs)
history = model.fit(train_ds, validation_data=val_ds, epochs=20)

# ✅ Save model safely
model.save("animal_classifier_model.keras")
print("🎉 Model saved successfully as 'animal_classifier_model.keras'")
