import librosa
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras.layers import Conv2DTranspose as Deconv2D
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
#multiply
from tensorflow.keras.layers import Multiply
import librosa.display
import matplotlib
import random


from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, Dropout, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import BatchNormalization, LeakyReLU


SAMPLE_RATE=8192
WINDOW_SIZE=1023
HOP_LENGTH=768




def batchV2(mus, nb_batch):
    #on choisit une piste aléatoire
    track = random.choice(mus.tracks)

    #on prends la piste et les vocals
    x = track.audio.T[0]
    y = track.targets['vocals'].audio.T[0]

    #avec librosa ==> resample
    x = librosa.resample(x,orig_sr=44100,target_sr=8192)
    y = librosa.resample(y,orig_sr=44100,target_sr=8192)

    #fft avec librosa
    X = librosa.stft(x, n_fft=WINDOW_SIZE)
    Y = librosa.stft(y, n_fft=WINDOW_SIZE)

    #on prends l'amplitude
    X_amp = librosa.amplitude_to_db(abs(X))
    Y_amp = librosa.amplitude_to_db(abs(Y))

    mix = []
    vocals = []

    START = random.randint(0, X_amp.shape[1]-128)

    for i in range(nb_batch):

        if START+128 >= X_amp.shape[1]-128:
            START = 0
        else:
            START += 20

        mix.append(X_amp[:,START:START+128])
        vocals.append(Y_amp[:,START:START+128])
    mix = np.array(mix)
    vocals = np.array(vocals)

    return mix, vocals





def plot_spectrogramme(spectrogramme) :
  plt.figure(figsize=(14,2))
  plt.title("Spectrogramme")
  print("plot le .T")
  librosa.display.specshow(spectrogramme.T,x_axis='linear',cmap= matplotlib.cm.jet)
  plt.xlabel("Temps")
  plt.ylabel("Fréquence")
  plt.colorbar()
  plt.show()



def unet(inputs=Input((512, 128, 1))):
    conv1 = Conv2D(16, 5, strides=2, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    conv2 = Conv2D(32, 5, strides=2, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    conv3 = Conv2D(64, 5, strides=2, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)

    conv4 = Conv2D(128, 5, strides=2, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)

    conv5 = Conv2D(256, 5, strides=2, padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)

    conv6 = Conv2D(512, 5, strides=2, padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.2)(conv6)

    deconv7 = Deconv2D(256, 5, strides=2, padding='same')(conv6)
    deconv7 = BatchNormalization()(deconv7)
    deconv7 = Dropout(0.5)(deconv7)
    deconv7 = Activation('relu')(deconv7)

    deconv8 = Concatenate(axis=3)([deconv7, conv5])
    deconv8 = Deconv2D(128, 5, strides=2, padding='same')(deconv8)
    deconv8 = BatchNormalization()(deconv8)
    deconv8 = Dropout(0.5)(deconv8)
    deconv8 = Activation('relu')(deconv8)

    deconv9 = Concatenate(axis=3)([deconv8, conv4])
    deconv9 = Deconv2D(64, 5, strides=2, padding='same')(deconv9)
    deconv9 = BatchNormalization()(deconv9)
    deconv9 = Dropout(0.5)(deconv9)
    deconv9 = Activation('relu')(deconv9)

    deconv10 = Concatenate(axis=3)([deconv9, conv3])
    deconv10 = Deconv2D(32, 5, strides=2, padding='same')(deconv10)
    deconv10 = BatchNormalization()(deconv10)
    deconv10 = Activation('relu')(deconv10)

    deconv11 = Concatenate(axis=3)([deconv10, conv2])
    deconv11 = Deconv2D(16, 5, strides=2, padding='same')(deconv11)
    deconv11 = BatchNormalization()(deconv11)
    deconv11 = Activation('relu')(deconv11)

    deconv12 = Concatenate(axis=3)([deconv11, conv1])
    deconv12 = Deconv2D(1, 5, strides=2, padding='same')(deconv12)
    deconv12 = Activation('relu')(deconv12)

    #en sortie, on a un mask de la meme taille que le spectrogramme d'entrée
    #il faut donc le multiplier par le spectrogramme d'entrée pour avoir le spectrogramme de sortie


    x = Multiply()([deconv12, inputs])

    #x = Conv2D(1, kernel_size=(5, 5), strides=1, activation='sigmoid', padding='same')(x)



    model = Model(inputs=inputs, outputs=x)
    return model
  



 



def affiche(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



def batchV2(mus, nb_batch):

    """
    Le but est de 
    1- prendre une piste au hasard dans la base de données
    2- demarrer a un point au hasard dans la piste
    3- prendre 128 trames de la piste
    4- Puis decaler de overlap trames et reprendre 128 trames .. jusqua a voir BATCH_SIZE trames
    5- extraire les spectrogrammes mix et vocals

    En sortie on a deux listes de taille BATCH_SIZE
    mix = [spectrogramme1, spectrogramme2, ... , spectrogrammeBATCH_SIZE]
    vocals = [spectrogramme1, spectrogramme2, ... , spectrogrammeBATCH_SIZE]

    avec spectrogramme1 contenants 128 trames de la piste 1 
    """



    #on choisit une piste aléatoire
    track = random.choice(mus.tracks)

    #on prends la piste et les vocals
    x = track.audio.T[0]
    y = track.targets['vocals'].audio.T[0]


    #avec librosa ==> resample
    x = librosa.resample(x,orig_sr=44100,target_sr=8192)
    y = librosa.resample(y,orig_sr=44100,target_sr=8192)

    #fft avec librosa
    X = librosa.stft(x, n_fft=WINDOW_SIZE)
    Y = librosa.stft(y, n_fft=WINDOW_SIZE)

    #on prends l'amplitude
    X_amp = librosa.amplitude_to_db(abs(X))
    Y_amp = librosa.amplitude_to_db(abs(Y))

    #on prends les frames

    mix = []
    vocals = []

    START = random.randint(0, X_amp.shape[1]-128)

    for i in range(nb_batch):

        if START+128 >= X_amp.shape[1]-128:
            START = 0
        else:
            START += 128

     
        mix.append(X_amp[:,START:START+128])
        vocals.append(Y_amp[:,START:START+128])
        
        

    mix = np.array(mix)
    vocals = np.array(vocals)

    return mix, vocals





def unetV2():
    # Partie encodeur
    inputs = Input(shape=(512, 128, 1))
    x = inputs
    conv1 = Convolution2D(16, kernel_size=(5, 5), padding='same', strides=2)(x)
    x = BatchNormalization()(conv1)
    x = LeakyReLU(alpha=0.2)(x)
    conv2 = Convolution2D(32, kernel_size=(5, 5), padding='same', strides=2)(x)
    x = BatchNormalization()(conv2)
    x = LeakyReLU(alpha=0.2)(x)
    conv3 = Convolution2D(64, kernel_size=(5, 5), padding='same', strides=2)(x)
    x = BatchNormalization()(conv3)
    x = LeakyReLU(alpha=0.2)(x)
    conv4 = Convolution2D(128, kernel_size=(5, 5), padding='same', strides=2)(x)
    x = BatchNormalization()(conv4)
    x = LeakyReLU(alpha=0.2)(x)
    conv5 = Convolution2D(256, kernel_size=(5, 5), padding='same', strides=2)(x)
    x = BatchNormalization()(conv5)
    x = LeakyReLU(alpha=0.2)(x)
    conv6 = Convolution2D(512, kernel_size=(5, 5), padding='same', strides=2)(x)
    x = BatchNormalization()(conv6)
    x = LeakyReLU(alpha=0.2)(x)

        # Partie décodeur
    x = Conv2DTranspose(256, kernel_size=(5, 5), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    skip1 = conv5  # Save the reference to conv5 for later multiplication
    x = concatenate([x, conv5], axis=-1)  # Skip connection
    x = Dropout(0.5)(x)
    x = Conv2DTranspose(128, kernel_size=(5, 5), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    skip2 = conv4  # Save the reference to conv4 for later multiplication
    x = concatenate([x, conv4], axis=-1)  # Skip connection
    x = Dropout(0.5)(x)
    x = Conv2DTranspose(64, kernel_size=(5, 5), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    skip3 = conv3  # Save the reference to conv3 for later multiplication
    x = concatenate([x, conv3], axis=-1)  # Skip connection
    x = Dropout(0.5)(x)
    x = Conv2DTranspose(32, kernel_size=(5, 5), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    skip4 = conv2  # Save the reference to conv2 for later multiplication
    x = concatenate([x, conv2], axis=-1)  # Skip connection
    x = Conv2DTranspose(16, kernel_size=(5, 5), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    skip5 = conv1  # Save the reference to conv1 for later multiplication
    x = concatenate([x, conv1], axis=-1)  # Skip connection
    x = Conv2DTranspose(1, kernel_size=(5, 5), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Multiply layer
    x = Multiply()([x, inputs])  # Multiply x with the feature maps from conv1
    x = Convolution2D(1, kernel_size=(5, 5), strides=1, activation='sigmoid', padding='same')(x)

    outputs = x
    unet = Model(inputs, outputs)
    unet.summary()

    return unet


def divide_into_frames(audio):
    spec_to_predict = []
    #fft
    y = librosa.stft(audio, n_fft=WINDOW_SIZE)
    print("shape de y : ",y.shape)

    for i in range(0,y.shape[1]-128,128):
        tmp = y[:,i:i+128]
        tmp = tmp.reshape(512,128,1)
        spec_to_predict.append(tmp)
    
    print("shape de la spec to predict : ",np.array(spec_to_predict).shape)
    
    return spec_to_predict



def reconstruct_from_frames(y,predicted_frames):
    # via isfft et la phase, on reconstruit le signal : voix et musique
    son_reconstruit = []

    #spec original
    module_y, phase_y = librosa.magphase(librosa.stft(y,n_fft=WINDOW_SIZE))


    for i in range(len(predicted_frames)):
        #reshape
     
        tmp = librosa.istft(predicted_frames[i]*phase_y[:,i:i+128])
        son_reconstruit.append(tmp)

    #reshape pour ne former qu'un seul vecteur
    son_reconstruit = np.array(son_reconstruit)
    son_reconstruit = son_reconstruit.reshape(son_reconstruit.shape[0]*son_reconstruit.shape[1])

    return son_reconstruit