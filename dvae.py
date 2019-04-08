from keras.layers import Lambda, Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from generator import DataGenerator
from config import Config
import h5py
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VAE():
    def __init__(self, wdir, ddir, latent_size, beta):
        self.wdir = wdir
        self.ddir = ddir
        self.image_size = 64
        self.batch_size = 64
        filters = 64

        if not os.path.exists(wdir):
            os.makedirs(wdir)

        x = Input(shape=(self.image_size, self.image_size, 3))
        conv1 = Conv2D(3, kernel_size=(2, 2), padding='same', activation='relu')(x)
        conv2 = Conv2D(filters, kernel_size=(2,2), padding='same', activation='relu', strides=(2, 2))(conv1)
        conv3 = Conv2D(filters, kernel_size=3, padding='same', activation='relu', strides=1)(conv2)
        conv4 = Conv2D(filters, kernel_size=3, padding='same', activation='relu', strides=1)(conv3)
        flat = Flatten()(conv4)

        z_mean = Dense(latent_size)(flat)
        z_std = Dense(latent_size)(flat)

        def sampling(args):
            z_mean, z_std = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_size), mean=0, stddev=1)
            return z_mean + K.exp(z_std) * epsilon

        z = Lambda(sampling, output_shape=(latent_size,))([z_mean, z_std])

        up = Dense(filters * (self.image_size // 2) * (self.image_size // 2), activation='relu')
        shape = (self.batch_size, self.image_size // 2, self.image_size // 2, filters)
        reshape = Reshape(shape[1:])
        deconv1 = Conv2DTranspose(filters, kernel_size=3, padding='same', strides=1, activation='relu')
        deconv2 = Conv2DTranspose(filters, kernel_size=3, padding='same', strides=1, activation='relu')
        shape = (self.batch_size, filters, self.image_size + 1, self.image_size + 1)
        deconv3 = Conv2DTranspose(filters, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu')
        x_reconstr = Conv2D(3, kernel_size=2, padding='valid', activation='sigmoid')

        up_ = up(z)
        reshape_ = reshape(up_)
        deconv1_ = deconv1(reshape_)
        deconv2_ = deconv2(deconv1_)
        deconv3_ = deconv3(deconv2_)
        x_reconstr_ = x_reconstr(deconv3_)

        self.vae = Model(x, x_reconstr_)

        xent_loss = self.image_size * self.image_size * binary_crossentropy(K.flatten(x), K.flatten(x_reconstr_))
        kl_loss = -.5 * K.sum(1 + z_std - K.square(z_mean) - K.exp(z_std), axis=-1)
        vae_loss = K.mean(xent_loss + beta * kl_loss)
        self.vae.add_loss(vae_loss)

        self.vae.compile(optimizer=Adam(lr=1e-3))
        self.vae.summary()

        self.encoder = Model(x, z_mean)

        _z = Input(shape=(latent_size,))
        _up = up(_z)
        _reshape = reshape(_up)
        _deconv1 = deconv1(_reshape)
        _deconv2 = deconv2(_deconv1)
        _deconv3 = deconv3(_deconv2)
        _x_reconstr = x_reconstr(_deconv3)

        self.decoder = Model(_z, _x_reconstr)

    def load_weights(self):
        try:
            self.vae.load_weights(self.wdir + '/weights.h5')
            print('Weights loaded')
            return True
        except:
            print('Failed to load weights')
            return False

    def train(self):
        self.load_weights()
        train_gen = DataGenerator(self.ddir + '/train', self.image_size, self.batch_size, train=True)
        dev_gen = DataGenerator(self.ddir + '/dev', self.image_size, self.batch_size)
        checkpoint_callback = ModelCheckpoint(os.path.join(self.wdir, 'weights.h5'), save_best_only=True, verbose=1)
        earlystopping_callback = EarlyStopping(verbose=1, patience=5)
        callbacks = [checkpoint_callback, earlystopping_callback]
        self.vae.fit_generator(train_gen, validation_data=dev_gen, epochs=999, shuffle='batch', callbacks=callbacks, verbose=1)

    def encode(self):
        self.load_weights()
        for type in ['test', 'dev', 'train']:
            print('Encoding {0}'.format(type))
            dpath = os.path.join(self.ddir, type)
            spath = os.path.join(self.wdir, type + '_encodings.h5')
            gen = DataGenerator(dpath, self.image_size, self.batch_size)
            z = self.encoder.predict_generator(gen, verbose=1)
            class_dict = {v: k for k, v in gen.generator.class_indices.items()}
            labels = [class_dict[x] for x in gen.generator.classes]
            with h5py.File(spath, 'w') as f:
                f.create_dataset('encodings', data=z)
                f.create_dataset('filenames', data=np.array(gen.generator.filenames, dtype='S'))
                f.create_dataset('labels', data=np.array(labels, dtype='S'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wdir', help='working directory', type=str, required=True)
    parser.add_argument('--ddir', help='data directory', type=str, default='/data/nlp/zap50k')
    parser.add_argument('--latent_size', type=int, default=2)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    model = VAE(args.wdir, args.ddir, args.latent_size, args.beta)
    if args.train:
        model.train()
    model.encode()

    #Save config for use in other scripts
    config = Config(args.wdir, args.ddir, args.latent_size, args.beta)
    config.save(args.wdir + '/config.json')
