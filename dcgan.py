from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
from tensorflow import keras
import glob
import imageio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import sys
import cv2

TRAIN = False
if len(sys.argv) == 2 and sys.argv[1] == 'train':
    TRAIN = True


'''
input_dir='/home/yiping.tang/all'
input_folder=os.listdir(input_dir)
num_input=len(input_folder)

image_cha=[]
image_cha_pa=[]
image_cha_class=[]
image_ora=[]
image_ora_pa=[]

for i in range(num_input):
#for i in range(1):
    if 1==0:
      seg_cha_path=os.path.join(input_dir,input_folder[i],'characters')
      img_class_cha=os.listdir(seg_cha_path)
      for ii in range(len(img_class_cha)):
        img_cha_class_pa=os.path.join(seg_cha_path,img_class_cha[ii])
        img_cha_path=os.listdir(img_cha_class_pa)
        for iii in range(len(img_cha_path)):
          img_cha_pa=os.path.join(img_cha_class_pa,img_cha_path[iii])
          im_cha=cv2.imread(img_cha_pa)  
          #image_cha.append(im_cha)
          image_cha_pa.append(img_cha_pa)
          image_cha_class.append(img_class_cha[ii])
    if 1==1:
      seg_ora_path=os.path.join(input_dir,input_folder[i],'images')
      img_all_ora=os.listdir(seg_ora_path)
      for ii in range(len(img_all_ora)):
        img_ora_pa=os.path.join(seg_ora_path,img_all_ora[ii])
        im_ora=cv2.imread(img_ora_pa)  
        im_ora_r=cv2.resize(im_ora,(1000,1500))
        image_ora.append(im_ora_r)
        image_ora_pa.append(img_ora_pa)
        papa=os.path.join('/home/yiping.tang/all_',img_all_ora[ii])
        cv2.imwrite(papa,im_ora_r)

'''

#train_images=image_ora
(train_images, _), (_, _) = keras.datasets.mnist.load_data()

""" 
input_path='/home/yiping.tang/full_kuzushiji'
input_list=os.listdir(input_path)
train_=[]
for i in range(len(input_list)):
  input_list_pa=os.path.join(input_path,input_list[i])
  im=cv2.imread(input_list_pa)
  img=cv2.resize(im,(1000,1500))
  train_.append(img) 
  """



(train_images, _), (_, _) = keras.datasets.fashion_mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)  
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)    
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model


generator = make_generator_model()


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model


discriminator = make_discriminator_model()


noise = tf.random.normal([1, 100])

generated_image = generator(noise, training=False)
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')

decision = discriminator(generated_image)

print(decision)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_dir = 'dcgan_training_checkpoints_fullpage_kuzushiji'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16


seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



def train(dataset, epochs,path):  
    for epoch in range(epochs+1):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        print("=======================================")
        generate_and_save_images(
            generator,
            epoch + 1,
            seed,
            path)
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



def generate_and_save_images(model, epoch, test_input,path):
    predictions = model.predict(test_input, steps=1)
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(os.path.join(path,'image_at_epoch_{:04d}_modify.png'.format(epoch)))
    # plt.show()
    plt.close()




def write_gif(path):
    anim_file = 'dcgan.gif'
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(path,'image*.png'))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


#path='c:/2019codh11/'
path='./generator'
path_k='./generator_k'
generate_and_save_images(
        generator,
        0000,
        seed,
        path)
        

TRAIN=True
if TRAIN:
    train(train_dataset, EPOCHS,path)
    write_gif(path)
else:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("After training:")
    generated_image = generator(noise, training=False)
    decision = discriminator(generated_image)
    print(decision)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    generate_and_save_images(
            generator,
            9999,
            seed,
            path_k)
