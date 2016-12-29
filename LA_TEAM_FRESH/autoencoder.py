import matplotlib.pyplot as plt
import random
from tqdm import tqdm_notebook as tqdm
from keras.layers import Dense, BatchNormalization, Activation
from keras.layers import UpSampling2D, Convolution2D, Input, Reshape
from keras.layers import LeakyReLU, Dropout, Flatten, GaussianNoise
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras import backend as K
print(K.image_dim_ordering())
import numpy as np
np.random.seed(42)

# Build Generative model ...
encoding_dim = 32

input_img = Input(shape=(784, ))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)
autoencoder.summary()

encoder = Model(input=input_img, output=encoded)

encoded_input = Input(shape=(encoding_dim, ))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.
print(X_train.shape, X_test.shape)
X_train = X_train.reshape((X_train.shape[0], np.prod(X_train.shape[1:])))
X_test = X_test.reshape((X_test.shape[0], np.prod(X_test.shape[1:])))
print(X_train.shape, X_test.shape)

autoencoder.fit(X_train, X_train, nb_epoch=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(X_test)
print(encoded_imgs.shape)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

