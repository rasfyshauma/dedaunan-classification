# dedaunan-classification
Anggota Kelompok :
Rasfy Shauma Moisa Asda (201910370311445)
Kurnia Putera Bagaskara (201910370311422)

# Bagian untuk cek dataset
```
import os

train_dir = "/content/gdrive/MyDrive/Colab Notebooks/Tubes AI/splitted/train"
val_dir = "/content/gdrive/MyDrive/Colab Notebooks/Tubes AI/splitted/test"

#train
belimbingwuluh_train_path = train_dir + '/Belimbing Wuluh'
jambubiji_train_path = train_dir + '/Jambu biji'
jeruk_train_path = train_dir + '/Jeruk'
kemangi_train_path = train_dir + '/Kemangi'
lidahbuaya_train_path = train_dir + '/Lidah buaya'
nangka_train_path = train_dir + '/Nangka'
pandan_train_path = train_dir + '/Pandan'
pepaya_train_path = train_dir + '/Pepaya'
seledri_train_path = train_dir + '/Seledri'
sirih_train_path = train_dir + '/Sirih'

#val
belimbingwuluh_val_path = val_dir + '/Belimbing Wuluh'
jambubiji_val_path = val_dir + '/Jambu biji'
jeruk_val_path = val_dir + '/Jeruk'
kemangi_val_path = val_dir + '/Kemangi'
lidahbuaya_val_path = val_dir + '/Lidah buaya'
nangka_val_path = val_dir + '/Nangka'
pandan_val_path = val_dir + '/Pandan'
pepaya_val_path = val_dir + '/Pepaya'
seledri_val_path = val_dir + '/Seledri'
sirih_val_path = val_dir + '/Sirih'

#jumlah train
belimbingwuluh_len_train = len(os.listdir(belimbingwuluh_train_path))
jambubiji_len_train = len(os.listdir(jambubiji_train_path))
jeruk_len_train = len(os.listdir(jeruk_train_path))
kemangi_len_train = len(os.listdir(kemangi_train_path))
lidahbuaya_len_train = len(os.listdir(lidahbuaya_train_path))
nangka_len_train = len(os.listdir(nangka_train_path))
pandan_len_train = len(os.listdir(pandan_train_path))
pepaya_len_train = len(os.listdir(pepaya_train_path))
seledri_len_train = len(os.listdir(seledri_train_path))
sirih_len_train = len(os.listdir(sirih_train_path))

#jumlah val
belimbingwuluh_len_val = len(os.listdir(belimbingwuluh_val_path))
jambubiji_len_val = len(os.listdir(jambubiji_val_path))
jeruk_len_val = len(os.listdir(jeruk_val_path))
kemangi_len_val = len(os.listdir(kemangi_val_path))
lidahbuaya_len_val = len(os.listdir(lidahbuaya_val_path))
nangka_len_val = len(os.listdir(nangka_val_path))
pandan_len_val = len(os.listdir(pandan_val_path))
pepaya_len_val = len(os.listdir(pepaya_val_path))
seledri_len_val = len(os.listdir(seledri_val_path))
sirih_len_val = len(os.listdir(sirih_val_path))

print("Dataset Training : ", belimbingwuluh_len_train + jambubiji_len_train 
      + jeruk_len_train + kemangi_len_train + lidahbuaya_len_train + nangka_len_train +
      pandan_len_train + pepaya_len_train + seledri_len_train + sirih_len_train)
print("Dataset Validasi : ", belimbingwuluh_len_val + jambubiji_len_val 
      + jeruk_len_val + kemangi_len_val + lidahbuaya_len_val + nangka_len_val +
      pandan_len_val + pepaya_len_val + seledri_len_val + sirih_len_val)
```

# Augmentasi
```
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
```
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')
```

# Arsitektur dan Fit Model
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, AveragePooling2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D
```
```
model = Sequential()

model.add(InputLayer(input_shape=[150,150,3]))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=2, padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dropout(0.0001))
```
```
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.0001))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.0001))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.0001))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.0001))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.0001))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.0001))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.0001))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.0001))
model.add(Dense(10, activation='sigmoid'))
```

# Compile Model
```
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['acc'])
```
```
history = model.fit(
      train_generator,
      steps_per_epoch=10,  # images = batch_size * steps
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50,  #  images = batch_size * steps
      )
```

# Evaluasi Model
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```
