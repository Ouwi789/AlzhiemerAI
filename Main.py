import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomZoom(0.2),
    layers.RandomWidth(0.2),
    layers.RandomHeight(0.2),
])

preprocess_input = tf.keras.Sequential([
    layers.Resizing(128, 128),
    layers.Rescaling(1./255) 
])

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory='/Path/To/Training/Data', 
    image_size=(248, 496),      
    batch_size=16,             
    label_mode='binary',
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory='/Path/To/Validation/Data', 
    image_size=(248, 496),      
    batch_size=16,              
    label_mode='binary'        
)

train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)

augmented_train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y), 
    num_parallel_calls=tf.data.AUTOTUNE
)

AUTOTUNE = tf.data.AUTOTUNE

augmented_train_dataset = augmented_train_dataset.shuffle(1000).take(1000).cache().repeat().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(buffer_size=AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(16, (3, 3), activation='leaky_relu'),
    layers.SpatialDropout2D(0.5),
    layers.Conv2DTranspose(32, (3, 3), activation='leaky_relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='leaky_relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2DTranspose(64, (3, 3), activation='leaky_relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='leaky_relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='leaky_relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.BinaryFocalCrossentropy(),
              metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='val_accuracy', mode='min', verbose = 1, patience = 15)

#adjust these through the number of items in the data files divided by your batch size
epochSteps = 21907 // 16
valSteps = 5544 // 16

model.fit(
    augmented_train_dataset,
    validation_data=val_dataset,
    epochs=25,
    callbacks=[earlyStopping],
    steps_per_epoch = epochSteps,
    validation_steps = valSteps,
)

model.save('/Path/To/Save/Model/model1.keras')

#summary of the validation accuracy and loss of the saved model
loss, accuracy = model.evaluate(val_dataset)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
print(f'Loss: {loss}')