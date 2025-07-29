# src/make_model_densenet.py

import os
import shutil
import sys
from keras.applications import DenseNet121
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import de la fonction split_data (doit être dans src/make_dataset.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from make_dataset import split_data


def build_densenet_model(config, data, labels, input_shape=(128, 128, 3)):
    """
    Construit un modèle DenseNet121 pré-entraîné et l'adapte au nombre de classes.
    """

    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Ajout de couches personnalisées
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)
    preds = Dense(config['num_classes'], activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=preds)
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall', 'AUC']
    )

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(data, labels)
    return model, x_train, x_val, x_test, y_train, y_val, y_test


def train_densenet_model(model, config, x_train, y_train, x_val, y_val, model_path="models/densenet"):
    """
    Entraîne le modèle avec data augmentation et sauvegarde localement le meilleur modèle.
    """

    # Créer le répertoire s’il n’existe pas
    os.makedirs(model_path, exist_ok=True)

    # Callbacks
    anne = ReduceLROnPlateau(
        monitor=config['monitor'],
        factor=config['factor'],
        patience=config['patience'],
        verbose=True,
        min_lr=config['min_lr']
    )
    checkpoint_path = os.path.join(model_path, 'model_best.keras')
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)

    # Data Augmentation
    datagen = ImageDataGenerator(
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.2
    )
    datagen.fit(x_train)

    # Entraînement
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=config['batch_size']),
        steps_per_epoch=x_train.shape[0] // config['batch_size'],
        epochs=config['epochs'],
        validation_data=(x_val, y_val),
        callbacks=[anne, checkpoint],
        verbose=2
    )

    if os.path.exists(checkpoint_path):
        print(f"✅ Modèle sauvegardé localement à : {checkpoint_path}")
    else:
        print("❌ Modèle non trouvé après l'entraînement")

    # Récapitulatif des paramètres
    training_params = {
        'batch_size': config['batch_size'],
        'epochs': config['epochs'],
        'steps_per_epoch': x_train.shape[0] // config['batch_size'],
        'learning_rate': model.optimizer.learning_rate.numpy(),
        'monitor': config['monitor'],
        'factor': config['factor'],
        'patience': config['patience'],
        'min_lr': config['min_lr']
    }

    return history, training_params
