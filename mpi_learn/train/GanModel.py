#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
import keras
from keras.models import Model
from keras.layers import Input
from keras import optimizers
from keras.optimizers import RMSprop,SGD
#from EcalEnergyGan import generator, discriminator
import numpy as np
import time
import socket

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling3D, Conv3D, ZeroPadding3D,
                                        AveragePooling3D)

from mpi_learn.utils import get_device_name
from mpi_learn.train.model import MPIModel, ModelBuilder

def hn():
    return socket.gethostname()


def weights(m):
    _weights_names = []
    for layer in m.layers:
        _weights_names += [ll.name for ll in layer.weights]
    _weights = m.get_weights()
    _disp = [(np.min(s),np.max(s),np.mean(s),np.std(s),s.shape,n) for s,n in zip(_weights,_weights_names)]
    for ii,dd in enumerate(_disp):
        print (ii,dd)

def weights_diff( m ,lap=True, init=False,label='', alert=1000.):
    if (weights_diff.old_weights is None) or init:
        weights_diff.old_weights = m.get_weights()
        return 
    _weights_names = []
    for layer in m.layers:
        _weights_names += [ll.name for ll in layer.weights]

    check_on_weight = m.get_weights()
    and_check_on_weight = weights_diff.old_weights
    ## make the diffs
    _diffs = [np.subtract(a,b) for (a,b) in zip(check_on_weight,and_check_on_weight)]
    _diffsN = [(np.min(s),np.max(s),np.mean(s),np.std(s),s.shape,n) for s,n in zip(_diffs,_weights_names)]
    #print ('\n'.join(['%s'%dd for dd in _diffsN]))
    for ii,dd in enumerate(_diffsN):
        if alert:
            if not any([abs(vv) > alert for vv in dd[:3]]):
                continue
        print (ii,'WD %s'%label,dd)
        #if dd[-2] == (8,):
        #    print ("\t",_diffs[ii])
    if lap:
        weights_diff.old_weights = m.get_weights()
        
weights_diff.old_weights = None

def discriminator(fixed_bn = False):

    image = Input(shape=( 25, 25, 25,1 ), name='image')

    bnm=2 if fixed_bn else 0
    x = Conv3D(32, 5, 5,5,border_mode='same',
               name='disc_c1')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2,2))(x)
    x = Conv3D(8, 5, 5,5,border_mode='valid',
               name='disc_c2'
    )(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(name='disc_bn1',
                           mode=bnm,
    )(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    x = Conv3D(8, 5, 5,5,border_mode='valid',
               name='disc_c3'
)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(name='disc_bn2',
                           #momentum = 0.00001
                           mode=bnm,
    )(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((1, 1, 1))(x)
    x = Conv3D(8, 5, 5,5,border_mode='valid',
               name='disc_c4'
    )(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(name='disc_bn3',
                           mode=bnm,
    )(x)
    x = Dropout(0.2)(x)

    x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h, name='dnn')

    dnn_out = dnn(image)

    fake = Dense(1, activation='sigmoid', name='classification')(dnn_out)
    aux = Dense(1, activation='linear', name='energy')(dnn_out)
    ecal = Lambda(lambda x: K.sum(x, axis=(1, 2, 3)))(image)

    return Model(output=[fake, aux, ecal], input=image, name='discriminator_model')

def generator(latent_size=200, return_intermediate=False):

    latent = Input(shape=(latent_size, ))

    bnm=0
    x = Dense(64 * 7* 7, init='glorot_normal',
              name='gen_dense1'
    )(latent)
    x = Reshape((7, 7,8, 8))(x)
    x = Conv3D(64, 6, 6, 8, border_mode='same', init='he_uniform',
               name='gen_c1'
    )(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(name='gen_bn1',
                           mode=bnm
    )(x)
    x = UpSampling3D(size=(2, 2, 2))(x)

    x = ZeroPadding3D((2, 2, 0))(x)
    x = Conv3D(6, 6, 5, 8, init='he_uniform',
               name='gen_c2'
    )(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(name='gen_bn2',
                           mode=bnm)(x)
    x = UpSampling3D(size=(2, 2, 3))(x)

    x = ZeroPadding3D((1,0,3))(x)
    x = Conv3D(6, 3, 3, 8, init='he_uniform',
               name='gen_c3')(x)
    x = LeakyReLU()(x)

    x = Conv3D(1, 2, 2, 2, bias=False, init='glorot_normal',
               name='gen_c4')(x)
    x = Activation('relu')(x)

    loc = Model(latent, x)
    fake_image = loc(latent)
    Model(input=[latent], output=fake_image)
    return Model(input=[latent], output=fake_image, name='generator_model')



def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


class GANModel(MPIModel):
    def __init__(self, latent_size=200, checkpoint=True):
        self.tell = True
        self._onepass = False#True
        self._heavycheck = True
        self._show_values = False
        self._show_loss = False
        self._show_weights = False
        self.latent_size=latent_size
        self.batch_size= None
        self.discr_loss_weights = [2, 0.1, 0.1]

        self.assemble_models()
        self.recompiled = False
        self.checkpoint = True
        
        if self.tell:
            print ("Generator summary")
            self.generator.summary()
            print ("Discriminator summary")
            self.discriminator.summary()
            print ("Combined summary")
            self.combined.summary()
        
        MPIModel.__init__(self, models = [
            self.discriminator,
            self.generator
        ])

        ## counters
        self.g_cc = 0
        self.d_cc = 0
        self.p_cc = 0
        self.g_t = []
        self.d_t = []
        self.p_t = []

        
    def big_assemble_models(self):

        image = Input(shape=( 25, 25, 25,1 ), name='image')

        x = Conv3D(32, 5, 5,5,border_mode='same')(image)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        x = ZeroPadding3D((2, 2,2))(x)
        x = Conv3D(8, 5, 5,5,border_mode='valid')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = ZeroPadding3D((2, 2, 2))(x)
        x = Conv3D(8, 5, 5,5,border_mode='valid')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = ZeroPadding3D((1, 1, 1))(x)
        x = Conv3D(8, 5, 5,5,border_mode='valid')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = AveragePooling3D((2, 2, 2))(x)
        h = Flatten()(x)
        

        fake = Dense(1, activation='sigmoid', name='generation')(h)
        aux = Dense(1, activation='linear', name='auxiliary')(h)
        ecal = Lambda(lambda x: K.sum(x, axis=(1, 2, 3)))(image)        

        self.discriminator = Model(output=[fake, aux, ecal], input=image, name='discriminator_model')

        latent = Input(shape=(self.latent_size, ))

        x = Dense(64 * 7* 7, init='he_uniform')(latent)
        x = Reshape((7, 7,8, 8))(x)
        x = Conv3D(64, 6, 6, 8, border_mode='same', init='he_uniform' )(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = UpSampling3D(size=(2, 2, 2))(x)

        x = ZeroPadding3D((2, 2, 0))(x)
        x = Conv3D(6, 6, 5, 8, init='he_uniform')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = UpSampling3D(size=(2, 2, 3))(x)
        
        x = ZeroPadding3D((1,0,3))(x)
        x = Conv3D(6, 3, 3, 8, init='he_uniform')(x)
        x = LeakyReLU()(x)
        
        x = Conv3D(1, 2, 2, 2, bias=False, init='glorot_normal')(x)
        x = Activation('relu')(x)
        
        loc = Model(latent, x)
        #fake_image = loc(latent)
        self.generator = Model(input=latent, output=x, name='generator_model')
        
        c_fake, c_aux, c_ecal = self.discriminator(x)
        self.combined = Model(
            input = latent,
            output = [c_fake, c_aux, c_ecal],
            name='combined_model'
            )
         
    
    def ext_assemble_models(self):
        print('[INFO] Building generator')
        self.generator = generator(self.latent_size)
        print('[INFO] Building discriminator')
        self.discriminator = discriminator()
        self.fixed_discriminator = discriminator(fixed_bn=True)
        print('[INFO] Building combined')
        #latent = Input(shape=(self.latent_size, ), name='combined_z')
        latent = Input(shape=(self.latent_size, ), name='combined_z')
        fake_image = self.generator(latent)
        #fake, aux, ecal = self.discriminator(fake_image)
        fake, aux, ecal = self.fixed_discriminator(fake_image)
        #self.discriminator.trainable = False        
        #fake, aux, ecal = self.discriminator( self.generator(latent))
        self.combined = Model(
            input=[latent],
            output=[fake, aux, ecal],
            name='combined_model'
        )

        #self.combined = Sequential(name='combined_model_seq')
        #self.combined.add(  self.generator)
        #self.combined.add( self.discriminator)

    def compile(self, **args):
        ## args are fully ignored here 
        print('[INFO] IN GAN MODEL: COMPILE')       

        def make_opt(**args):
            lr = args.get('lr',0.0001)
            prop = args.get('prop',True)
            if prop:
                opt = RMSprop()
            else:
                opt = SGD(lr=lr,
                #opt = VSGD(lr=lr,                          
                          #clipnorm = 1000.,
                           clipvalue = 1000.,
                )

            print ("optimizer for compiling",opt)
            return opt
        
        self.generator.compile(
            optimizer=make_opt(**args),
            loss='binary_crossentropy') ## never  actually used for training
        
        self.discriminator.compile(
            optimizer=make_opt(**args),            
            loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
            loss_weights=self.discr_loss_weights
        )

        if hasattr(self,'fixed_discriminator'):
            self.fixed_discriminator.trainable = False
        else:
            self.discriminator.trainable = False
        
        self.combined.compile(
            optimizer=make_opt(**args),
            loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
            loss_weights=self.discr_loss_weights
        ) 

        print ("compiled")

    def assemble_models(self):
        self.ext_assemble_models()
        
    def batch_transform(self, x, y):
        root_fit = [0.0018, -0.023, 0.11, -0.28, 2.21]
        x_disc_real =x
        y_disc_real =y
        show_values = self._show_values
        def mm( label, t):
            print (label,np.min(t),np.max(t),np.mean(t),np.std(t),t.shape)
        
        if self.batch_size is None:
            ## fix me, maybe
            self.batch_size = x_disc_real.shape[0]
            print (hn(),"initializing sizes",x_disc_real.shape,[ yy.shape for yy in y])


        noise = np.random.normal(0, 1, (self.batch_size, self.latent_size))
        sampled_energies = np.random.uniform(0.1, 5,(self.batch_size,1))
        generator_ip = np.multiply(sampled_energies, noise)
        #if show_values: print ('energies',np.ravel(sampled_energies)[:10])
        if show_values: mm('energies',sampled_energies)
        ratio = np.polyval(root_fit, sampled_energies)
        #if show_values: print ('ratios',np.ravel(ratio)[:10])
        if show_values: mm('ratios',ratio)
        ecal_ip = np.multiply(ratio, sampled_energies)
        #if show_values: print ('estimated sum cells',np.ravel(ecal_ip)[:10])
        if show_values: mm('estimated sum cells',ecal_ip)
        
        now = time.mktime(time.gmtime())
        if self.p_cc>1 and len(self.p_t)%100==0:
            print ("prediction average",np.mean(self.p_t),"[s]' over",len(self.p_t))
        generated_images = self.generator.predict(generator_ip)
        ecal_rip = np.squeeze(np.sum(generated_images, axis=(1, 2, 3)))
        #if show_values: print ('generated sum cells',np.ravel(ecal_rip)[:10])
        if show_values: mm('generated sum cells',ecal_rip)

        norm_overflow = False
        apply_identify = False ## False was intended originally
        
        if norm_overflow and np.max( ecal_rip ) > 1000.:
            if show_values: print ("normalizing back")
            #ecal_ip = ecal_rip
            generated_images /= np.max( generated_images )
            ecal_rip = np.squeeze(np.sum(generated_images, axis=(1, 2, 3)))
            #if show_values: print ('generated sum cells',np.ravel(ecal_rip)[:10])
            if show_values: mm('generated sum cells',ecal_rip)
        elif apply_identify:
            ecal_ip = ecal_rip
            
        done = time.mktime(time.gmtime())
        if self.p_cc:
            self.p_t.append( done - now )
        self.p_cc +=1

        ## need to bit flip the true labels too
        bf = bit_flip(y[0])
        y_disc_fake = [bit_flip(np.zeros(self.batch_size)), sampled_energies.reshape((-1,)), ecal_ip.reshape((-1,))]
        re_y = [bf, y_disc_real[1], y_disc_real[2]]

        if self._onepass:
            ## train the discriminator in one go
            bb_x  = np.concatenate( (x_disc_real, generated_images))
            bb_y = [np.concatenate((a,b)) for a,b in zip(re_y, y_disc_fake)]
            rng_state = np.random.get_state()
            np.random.shuffle( bb_x )
            np.random.set_state(rng_state)
            np.random.shuffle( bb_y[0] )
            np.random.set_state(rng_state)
            np.random.shuffle( bb_y[1] )
            np.random.set_state(rng_state)
            np.random.shuffle( bb_y[2] )        
        
            X_for_disc = bb_x
            Y_for_disc = bb_y



        c_noise = np.random.normal(0, 1, (2*self.batch_size, self.latent_size))
        ###print ('noise',np.ravel(noise)[:10])
        c_sampled_energies = np.random.uniform(0.1, 5, (2*self.batch_size,1 ))
        c_generator_ip = np.multiply(c_sampled_energies, c_noise)
        c_ratio = np.polyval(root_fit, c_sampled_energies)
        c_ecal_ip = np.multiply(c_ratio, c_sampled_energies)
        c_trick = np.ones(2*self.batch_size)        

        X_for_combined = [c_generator_ip]
        Y_for_combined = [c_trick,c_sampled_energies.reshape((-1, 1)), c_ecal_ip]

        if self._onepass:
            return (X_for_disc,Y_for_disc,X_for_combined,Y_for_combined)
        else:
            def head(a):
                return [o[:self.batch_size] for o in a]
            def tail(a):
                return [o[self.batch_size:] for o in a]            

            return ((x_disc_real,re_y),(generated_images,y_disc_fake), (head(X_for_combined),head(Y_for_combined)), (tail(X_for_combined),tail(Y_for_combined)))

    def test_on_batch(self,x, y, sample_weight=None):
        show_loss = self._show_loss
        if self._onepass:
            (X_for_disc,Y_for_disc,X_for_combined,Y_for_combined) = self.batch_transform(x,y)
            epoch_disc_loss = self.discriminator.test_on_batch(X_for_disc,Y_for_disc)
            epoch_gen_loss = self.combined.test_on_batch(X_for_combined,Y_for_combined)
        else:
            ((x_disc_real,re_y),(generated_images, y_disc_fake),(x_comb1,y_comb1),(x_comb2,y_comb2)) = self.batch_transform(x,y)
            real_disc_loss = self.discriminator.test_on_batch( x_disc_real,re_y )
            fake_disc_loss = self.discriminator.test_on_batch( generated_images, y_disc_fake)
            epoch_disc_loss = [(a + b) / 2 for a, b in zip(real_disc_loss, fake_disc_loss)]

            c_loss1= self.combined.test_on_batch( x_comb1,y_comb1 )
            c_loss2= self.combined.test_on_batch(x_comb2,y_comb2 )
            epoch_gen_loss = [(a + b) / 2 for a, b in zip(c_loss1,c_loss2)]
            if show_loss:
                print ("test discr loss",real_disc_loss,fake_disc_loss)
                print ("test combined loss",c_loss1, c_loss2)

            

        
        return np.asarray([epoch_disc_loss, epoch_gen_loss])

    def _onepass_train_on_batch(self, x, y,
                   sample_weight=None,
                   class_weight=None):

        (X_for_disc,Y_for_disc,X_for_combined,Y_for_combined) = self.batch_transform(x,y)
        if self.d_cc>1 and len(self.d_t)%100==0:
            print ("discriminator average",np.mean(self.d_t),"[s] over ",len(self.d_t))
        now = time.mktime(time.gmtime())
        epoch_disc_loss = self.discriminator.train_on_batch(X_for_disc,Y_for_disc)
        done = time.mktime(time.gmtime())
        if self.d_cc:
            self.d_t.append( done - now )
        self.d_cc+=1
        if self.g_cc>1 and len(self.g_t)%100==0:
            print ("generator average ",np.mean(self.g_t),"[s] over",len(self.g_t))
        now = time.mktime(time.gmtime())
        epoch_gen_loss = self.combined.train_on_batch(X_for_combined,Y_for_combined)
        done = time.mktime(time.gmtime())
        if self.g_cc:
            self.g_t.append( done - now )
            
        return np.asarray([epoch_disc_loss, epoch_gen_loss])
    
    def train_on_batch(self, x, y,
                   sample_weight=None,
                   class_weight=None):

        with np.errstate( divide='raise', invalid='raise' , over='raise', under ='raise' ):
            if self._onepass:
                return self._onepass_train_on_batch(x,y,sample_weight,class_weight)
            else:
                return self._twopass_train_on_batch(x,y,sample_weight,class_weight)

        
    def _twopass_train_on_batch(self, x, y,
                   sample_weight=None,
                   class_weight=None):

        ((x_disc_real,re_y),(generated_images, y_disc_fake),(x_comb1,y_comb1),(x_comb2,y_comb2)) = self.batch_transform(x,y)

        show_loss = self._show_loss
        show_weights = self._show_weights
        if self.d_cc>1 and len(self.d_t)%100==0:
            print ("discriminator average",np.mean(self.d_t),"[s] over ",len(self.d_t))
        self.discriminator.trainable = True

        if self._heavycheck:
            #on_weight = self.generator
            on_weight = self.combined
            check_on_weight = on_weight.get_weights()
            weights_names = []
            for l in on_weight.layers:
                weights_names += [ll.name for ll in l.weights]
            if self._show_weights:
                weights( on_weight )
            weights_diff( on_weight , init=True)
        
        now = time.mktime(time.gmtime())
        real_batch_loss = self.discriminator.train_on_batch(x_disc_real,re_y)
        if hasattr(self,'fixed_discriminator'):
            self.fixed_discriminator.set_weights( self.discriminator.get_weights())
        if self._heavycheck:
            if show_weights: weights( on_weight )
            weights_diff( on_weight , label='D-real')
        
        fake_batch_loss = self.discriminator.train_on_batch(generated_images, y_disc_fake)

        if hasattr(self,'fixed_discriminator'):
            ## pass things over
            self.fixed_discriminator.set_weights( self.discriminator.get_weights())
            self.fixed_discriminator.trainable = False
        else:
            self.discriminator.trainable = False
            
        if self._heavycheck:
            if show_weights: weights( on_weight )
            weights_diff( on_weight , label='D-fake')

            
        if show_loss:
            #print (self.discriminator.metrics_names)
            print (self.d_cc,"discr loss",real_batch_loss,fake_batch_loss)
        epoch_disc_loss = [(a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)]
        done = time.mktime(time.gmtime())
        if self.d_cc:
            self.d_t.append( done - now )
        self.d_cc+=1

        if show_weights:
            #print ("Disc",np.ravel(self.discriminator.get_weights()[1])[:10])
            #print ("Gen",np.ravel(self.generator.get_weights()[0])[:10])
            #print ("Comb",np.ravel(self.combined.get_weights()[0])[:10])
            weights( self.discriminator )
            weights( self.generator )
            weights( self.combined )
            



            #print ("Right",np.ravel(check_on_weight[1])[:10])

        if self.g_cc>1 and len(self.g_t)%100==0:
            print ("generator average ",np.mean(self.g_t),"[s] over",len(self.g_t))
            now = time.mktime(time.gmtime())

        epoch_gen_loss = []
        done = time.mktime(time.gmtime())
        if self.g_cc:
            self.g_t.append( done - now )
        c_loss1= self.combined.train_on_batch( x_comb1,y_comb1)
        if self._heavycheck:
            if show_weights: weights( on_weight )
            weights_diff( on_weight , label ='C-1')        
        c_loss2= self.combined.train_on_batch( x_comb2,y_comb2)
        #c_loss2= c_loss1
        if self._heavycheck:
            if show_weights: weights( on_weight )
            weights_diff( on_weight , label='C-2')
            
        if show_loss:
            #print(self.combined.metrics_names)
            print (self.g_cc,"combined loss",c_loss1,c_loss2)
        epoch_gen_loss = [(a + b) / 2 for a, b in zip(c_loss1,c_loss2)]
        done = time.mktime(time.gmtime())
        if self.g_cc:
            self.g_t.append( done - now )
        self.g_cc+=1

        if self._heavycheck:
            and_check_on_weight = on_weight.get_weights()
            
            ## this contains a boolean whether all values of the tensors are equal :
            # [--, False, --] ===> weights have changed for one layer
            # [--, True, --] ===> weights are the same one layer
            if show_weights:
                checks = [np.all(np.equal(a,b)) for (a,b) in zip(check_on_weight,and_check_on_weight)]
                weights_have_changed = not all(checks)
                weights_are_all_equal = all(checks)
                print ('Weights are the same?',checks)
                if weights_have_changed:
                    for iw,b in enumerate(checks):
                        if not b:
                            print (iw,"This",check_on_weight[iw].shape)
                            print (np.ravel(check_on_weight[iw])[:10])
                            print (iw,"And that",and_check_on_weight[iw].shape)
                            print (np.ravel(and_check_on_weight[iw])[:10])
                else:
                    print ("weights are all identical")
                    print (np.ravel(and_check_on_weight[1])[:10])
                    print (np.ravel(check_on_weight[1])[:10])

        ## checkpoints the models each time

        if self.checkpoint:
            import os            
            self.generator.save_weights('mpi_generator_%s.h5'%(os.getpid()))

        ## modify learning rate
        #switching_loss = (5.,5.)
        switching_loss = (1.,1.)
        if not self.recompiled and epoch_disc_loss[0]<switching_loss[0] and epoch_gen_loss[0]<switching_loss[1]:
            ## go on
            print ("going for full sgd")
            self.recompiled = True            
            self.compile( prop=False, lr=1.0)
            #K.set_value( self.discriminator.optimizer.lr, 1.0)
            #K.set_value( self.generator.optimizer.lr, 1.0)

            
        print (K.get_value(self.discriminator.optimizer.lr))
        #print (self.discriminator.optimizer.clipvalue)
        
        return np.asarray([epoch_disc_loss, epoch_gen_loss])
    

"""
        ## the rest is legacy
        root_fit = [0.0018, -0.023, 0.11, -0.28, 2.21]
        x_disc_real =x
        y_disc_real = y
        if self.batch_size is None:
            ## fix me, maybe 
            self.batch_size = x_disc_real.shape[0]
            print ("In train_on_batch: initializing sizes",x_disc_real.shape,[ yy.shape for yy in y])
            
            
        noise = np.random.normal(0, 1, (self.batch_size, self.latent_size))
        sampled_energies = np.random.uniform(0.1, 5,(self.batch_size,1))
        generator_ip = np.multiply(sampled_energies, noise)
        ratio = np.polyval(root_fit, sampled_energies)
        ecal_ip = np.multiply(ratio, sampled_energies)
        now = time.mktime(time.gmtime())
        if self.p_cc>1 and len(self.p_t)%100==0:
            print ("prediction average",np.mean(self.p_t),"[s]' over",len(self.p_t))
        generated_images = self.generator.predict(generator_ip)
        done = time.mktime(time.gmtime())
        if self.p_cc:
            self.p_t.append( done - now )
        self.p_cc +=1
        
        ## need to bit flip the true labels too
        bf = bit_flip(y[0])
        y_disc_fake = [bit_flip(np.zeros(self.batch_size)), sampled_energies.reshape((-1,)), ecal_ip.reshape((-1,))]
        re_y = [bf, y_disc_real[1], y_disc_real[2]]
        #print ("got",[yy.shape for yy in re_y])
        #print ("got",[yy.shape for yy in y_disc_fake])
        
        two_pass = True
        #print ("calling discr",self.d_cc)
        if self.d_cc>1 and len(self.d_t)%100==0:
            print ("discriminator average",np.mean(self.d_t),"[s] over ",len(self.d_t))
        now = time.mktime(time.gmtime())
        if two_pass:
      
            #real_batch_loss = self.discriminator.train_on_batch(x_disc_real, y_disc_real)
            real_batch_loss = self.discriminator.train_on_batch(x_disc_real,re_y)
            fake_batch_loss = self.discriminator.train_on_batch(generated_images, y_disc_fake)
            epoch_disc_loss = [
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ]
        else:
            ## train the discriminator in one go
            bb_x  = np.concatenate( (x_disc_real, generated_images))
            bb_y = [np.concatenate((a,b)) for a,b in zip(re_y, y_disc_fake)]
            rng_state = np.random.get_state()
            np.random.shuffle( bb_x )
            np.random.set_state(rng_state)
            np.random.shuffle( bb_y[0] )
            np.random.set_state(rng_state)
            np.random.shuffle( bb_y[1] )
            np.random.set_state(rng_state)
            np.random.shuffle( bb_y[2] )
            epoch_disc_loss = self.discriminator.train_on_batch( bb_x, bb_y )


        done = time.mktime(time.gmtime())
        if self.d_cc:
            self.d_t.append( done - now )
        self.d_cc+=1


        #print ("calling gen",self.g_cc)
        if self.g_cc>1 and len(self.g_t)%100==0:
            print ("generator average ",np.mean(self.g_t),"[s] over",len(self.g_t))
        now = time.mktime(time.gmtime())
        if two_pass:
            gen_losses = []
            trick = np.ones(self.batch_size)        
            for _ in range(2):
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_size))
                sampled_energies = np.random.uniform(0.1, 5, ( self.batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ratio = np.polyval(root_fit, sampled_energies)
                ecal_ip = np.multiply(ratio, sampled_energies)
                
                gen_losses.append(self.combined.train_on_batch(
                    [generator_ip],
                    [trick, sampled_energies.reshape((-1, 1)), ecal_ip]))

            epoch_gen_loss = [
                (a + b) / 2 for a, b in zip(*gen_losses)
            ]
        else:
            noise = np.random.normal(0, 1, (2*self.batch_size, self.latent_size))
            sampled_energies = np.random.uniform(0.1, 5, (2*self.batch_size,1 ))
            generator_ip = np.multiply(sampled_energies, noise)
            ratio = np.polyval(root_fit, sampled_energies)
            ecal_ip = np.multiply(ratio, sampled_energies)
            trick = np.ones(2*self.batch_size)
            epoch_gen_loss = self.combined.train_on_batch(
                [generator_ip],
                [trick,sampled_energies.reshape((-1, 1)), ecal_ip])

            
        
        done = time.mktime(time.gmtime())
        if self.g_cc:
            self.g_t.append( done - now )
        self.g_cc+=1
        
        #self.epoch_disc_loss.extend( epoch_disc_loss )
        #self.epoch_gen_loss.extend( epoch_gen_loss )
        return np.asarray([epoch_disc_loss, epoch_gen_loss])
"""


class GANModelBuilder(ModelBuilder):
    def __init__(self, c, device_name='cpu',tf=False):
        ModelBuilder.__init__(self, c)
        self.tf = tf
        self.device = self.get_device_name(device_name) if self.tf else None
        
    def build_model(self):
        if self.tf:
            ## I have to declare the device explictely
            #with K.tf.device('/gpu:0'):#self.device if 'gpu' in self.device else ''):
            #with K.tf.device(self.device if 'gpu' in self.device else ''):
            m = GANModel()
            return m
        else:
            m = GANModel()
            return m
            
    def get_device_name(self, device):
        """Returns a TF-style device identifier for the specified device.
            input: 'cpu' for CPU and 'gpuN' for the Nth GPU on the host"""
        if device == 'cpu':
            dev_num = 0
            dev_type = 'cpu'
        elif device.startswith('gpu'):
            try:
                dev_num = int(device[3:])
                dev_type = 'gpu'
            except ValueError:
                print ("GPU number could not be parsed from {}; using CPU".format(device))
                dev_num = 0
                dev_type = 'cpu'
        else:
            print ("Please specify 'cpu' or 'gpuN' for device name")
            dev_num = 0
            dev_type = 'cpu'
        return get_device_name(dev_type, dev_num, backend='tensorflow')