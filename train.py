import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim', type=int, default=100,
                       help='Noise dimension')

    parser.add_argument('--t_dim', type=int, default=256,
                       help='Text feature dimension')

    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')

    parser.add_argument('--image_size', type=int, default=24,
                       help='Image Size a, 7 x a')

    parser.add_argument('--gf_dim', type=int, default=64,
                       help='Number of conv in the first layer gen.')

    parser.add_argument('--df_dim', type=int, default=64,
                       help='Number of conv in the first layer discr.')

    parser.add_argument('--gfc_dim', type=int, default=1024,
                       help='Dimension of gen untis for for fully connected layer 1024')

    parser.add_argument('--caption_vector_length', type=int, default=88,
                       help='Caption Vector Length')

    parser.add_argument('--data_dir', type=str, default="Data",
                       help='Data Directory')

    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='Learning Rate')

    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Momentum for Adam Update')

    parser.add_argument('--epochs', type=int, default=600,
                       help='Max number of epochs')

    parser.add_argument('--save_every', type=int, default=30,
                       help='Save Model/Samples every x iterations over batches')

    parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

    parser.add_argument('--data_set', type=str, default="flowers",
                       help='Dat set: MS-COCO, flowers')

    args = parser.parse_args()
    model_options = {
        'z_dim' : args.z_dim,#noise dimension
        't_dim' : args.t_dim,#text feature dimension
        'batch_size' : args.batch_size,
        'image_size' : args.image_size,
        'gf_dim' : args.gf_dim,#first conv of generator
        'df_dim' : args.df_dim,#first conv of discriminator
        'gfc_dim' : args.gfc_dim,#dimension of fully connected layer
        'caption_vector_length' : args.caption_vector_length#length of caption vector
    }


    gan = model.GAN(model_options)
    input_tensors, variables, loss, outputs, checks = gan.build_model()

    d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
    g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    saver = tf.train.Saver()
    if args.resume_model:
        saver.restore(sess, args.resume_model)

    loaded_data = load_training_data(args.data_dir, args.data_set)

    for i in range(args.epochs):
        batch_no = 0
        while batch_no*args.batch_size < loaded_data['data_length']:
            real_images, wrong_images, caption_vectors, z_noise, image_files = get_training_batch(batch_no, args.batch_size,
                args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, loaded_data)

            # DISCR UPDATE
            check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
            _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
                feed_dict = {
                    input_tensors['t_real_image'] : real_images,
                    input_tensors['t_wrong_image'] : wrong_images,
                    input_tensors['t_real_caption'] : caption_vectors,
                    input_tensors['t_z'] : z_noise,
                })

            print "d1", d1
            print "d2", d2
            print "d3", d3
            print "D", d_loss

            # GEN UPDATE
            _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
                feed_dict = {
                    input_tensors['t_real_image'] : real_images,
                    input_tensors['t_wrong_image'] : wrong_images,
                    input_tensors['t_real_caption'] : caption_vectors,
                    input_tensors['t_z'] : z_noise,
                })

            # GEN UPDATE TWICE, to make sure d_loss does not go to 0
            _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
                feed_dict = {
                    input_tensors['t_real_image'] : real_images,
                    input_tensors['t_wrong_image'] : wrong_images,
                    input_tensors['t_real_caption'] : caption_vectors,
                    input_tensors['t_z'] : z_noise,
                })

            print "LOSSES", d_loss, g_loss, batch_no, i, len(loaded_data['image_list'])/ args.batch_size
            batch_no += 1
            if (batch_no % args.save_every) == 0:
                print "Saving Images, Model"#batch
                save_for_vis(args.data_dir, real_images, gen, image_files)
                save_path = saver.save(sess, "Data/Models/latest_model_{}_temp.ckpt".format(args.data_set))
        if i%5 == 0:#epoch
            save_path = saver.save(sess, "Data/Models/model_after_{}_epoch_{}.ckpt".format(args.data_set, i))


def load_img():
    img = np.load('/home/eve/traffic/y_img_train.npy').item()
    
    return img  # dict [link_id] = ndarray<7, 24, 1>

img = load_img()

def load_vector():
    vector = np.load('/home/eve/traffic/X_vector_train.npy').item()
    return vector  # dict [link_id] = ndarray<88,>

vector = load_vector()

def load_training_data(data_dir, data_set):
    if data_set == 'flowers':
        flower_captions = vector
        training_image_list = [key for key in vector]
        random.shuffle(training_image_list)

        return {
            'image_list' : training_image_list,#training data. contains the filenames of img / link_id
            'captions' : flower_captions,#captions: dict [filename of img] = caption_vector / spatial feature
            'data_length' : len(training_image_list)
        }


def save_for_vis(data_dir, real_images, generated_images, image_files):

    shutil.rmtree(join(data_dir, 'samples') )
    os.makedirs(join(data_dir, 'samples') )

    for i in range(0, real_images.shape[0]):
        real_image_255 = np.zeros( (7,24,1), dtype=np.uint8)
        real_images_255 = (real_images[i,:,:,:])
        scipy.misc.imsave( join(data_dir, 'samples/{}_{}.txt'.format(i, image_files[i].split('/')[-1] )) , real_images_255)

        fake_image_255 = np.zeros( (7,24,1), dtype=np.uint8)
        fake_images_255 = (generated_images[i,:,:,:])
        scipy.misc.imsave(join(data_dir, 'samples/fake_image_{}.txt'.format(i)), fake_images_255)


def get_training_batch(batch_no, batch_size, image_size, z_dim, 
    caption_vector_length, split, data_dir, data_set, loaded_data = None):

    real_images = np.zeros((batch_size, 7, 24, 1))
    wrong_images = np.zeros((batch_size, 7, 24, 1))
    captions = np.zeros((batch_size, caption_vector_length))
    
    # loaded_data
    # return {
    #     'image_list': training_image_list,  # training data. contains the filenames of img / link_id
    #     'captions': flower_captions,  # captions: dict [filename of img] = caption_vector / spatial feature
    #     'data_length': len(training_image_list)
    # }

    cnt = 0
    image_files = []
    for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
        idx = i % len(loaded_data['image_list'])
        image_file = loaded_data['image_list'][idx]
        # image_file = join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][idx])
        # image_array = image_processing.load_image_array(image_file, image_size)
        image_array = img[image_file]
        real_images[cnt,:,:,:] = image_array

        # Improve this selection of wrong image
        while(True):
            wrong_image_idx = random.randint(0,len(loaded_data['image_list'])-1)
            if(wrong_image_idx != idx):
                break
        # wrong_image_file = join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][wrong_image_id])
        wrong_image_file = loaded_data['image_list'][wrong_image_idx]
        # wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)
        wrong_image_array = img[wrong_image_file]
        wrong_images[cnt, :,:,:] = wrong_image_array
        
        
        cur_caption = loaded_data['captions'][ loaded_data['image_list'][idx] ]
        if(len(cur_caption) != caption_vector_length):
            print ('wrong !!!!!! error!!!!!!!')
        captions[cnt, :] = cur_caption
        
        # random_caption = random.randint(0,4)
        # captions[cnt,:] = loaded_data['captions'][ loaded_data['image_list'][idx] ][ random_caption ][0:caption_vector_length]
        image_files.append( image_file )
        cnt += 1

    z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
    return real_images, wrong_images, captions, z_noise, image_files

if __name__ == '__main__':
    main()
