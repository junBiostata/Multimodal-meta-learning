import os
import tensorflow as tf
import pandas as pd
import time
import random
import numpy as np
import sys
import argparse
import json

def get_batch_train(BATCH_SIZE, y_train, x_train_ct, x_train_hne, ystatus_train):
    while True:            
        j = 0
        while (j + 1) * BATCH_SIZE <= len(x_train_ct):
            x_batch_ct = x_train_ct[int(j * BATCH_SIZE):int((j + 1) * BATCH_SIZE), :]
            x_batch_hne = x_train_hne[int(j * BATCH_SIZE):int((j + 1) * BATCH_SIZE), :]
            y_batch = y_train[int(j * BATCH_SIZE):int((j + 1) * BATCH_SIZE)]
            ystatus_batch = ystatus_train[int(j * BATCH_SIZE):int((j + 1) * BATCH_SIZE)]
                
            x_batch_ct = np.array(x_batch_ct)
            x_batch_hne = np.array(x_batch_hne)
            y_batch = np.array(y_batch).reshape((-1, 1))
            ystatus_batch = np.array(ystatus_batch).reshape((-1, 1))
            
            j += 1
            yield x_batch_ct, x_batch_hne, y_batch, ystatus_batch

def get_batch_holdout(BATCH_SIZE, y_holdout, x_holdout_ct, x_holdout_hne, ystatus_holdout):
    while True:            
        j = 0
        while (j + 1) * BATCH_SIZE <= len(x_holdout_ct):
            x_batch_ct = x_holdout_ct[int(j * BATCH_SIZE):int((j + 1) * BATCH_SIZE), :]
            x_batch_hne = x_holdout_hne[int(j * BATCH_SIZE):int((j + 1) * BATCH_SIZE), :]
            y_batch = y_holdout[int(j * BATCH_SIZE):int((j + 1) * BATCH_SIZE)]
            ystatus_batch = ystatus_holdout[int(j * BATCH_SIZE):int((j + 1) * BATCH_SIZE)]
                
            x_batch_ct = np.array(x_batch_ct)
            x_batch_hne = np.array(x_batch_hne)
            y_batch = np.array(y_batch).reshape((-1, 1))
            ystatus_batch = np.array(ystatus_batch).reshape((-1, 1))
            
            j += 1
            yield x_batch_ct, x_batch_hne, y_batch, ystatus_batch
            
def train(y_holdout, x_holdout_ct, x_holdout_hne, ystatus_holdout, y_train, x_train_ct, x_train_hne, ystatus_train, checkpoint):
    
    np.set_printoptions(threshold=np.inf)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    
    regularizer = tf.keras.regularizers.l2(REG_SCALE)
    x_ct = tf.compat.v1.placeholder(tf.float32, [None, CT_FEATURE_SIZE], name='input_ct')
    x_hne = tf.compat.v1.placeholder(tf.float32, [None, HNE_FEATURE_SIZE], name='input_hne')
    ystatus = tf.compat.v1.placeholder(tf.float32, [None, 1], name='ystatus')
    R_matrix = tf.compat.v1.placeholder(tf.float32, [None, None], name='R_matrix')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    
    # CT Branch
    ct_dense_layer1 = tf.keras.layers.Dense(units=3000, activation=tf.nn.relu, kernel_regularizer=regularizer)(x_ct)
    ct_dense_drop1 = tf.nn.dropout(ct_dense_layer1, rate=1-keep_prob)
    ct_dense_layer2 = tf.keras.layers.Dense(units=1000, activation=tf.nn.relu, kernel_regularizer=regularizer)(ct_dense_drop1)
    ct_dense_drop2 = tf.nn.dropout(ct_dense_layer2, rate=1-keep_prob)
    
    # H&E Branch
    hne_dense_layer1 = tf.keras.layers.Dense(units=3000, activation=tf.nn.relu, kernel_regularizer=regularizer)(x_hne)
    hne_dense_drop1 = tf.nn.dropout(hne_dense_layer1, rate=1-keep_prob)
    hne_dense_layer2 = tf.keras.layers.Dense(units=1000, activation=tf.nn.relu, kernel_regularizer=regularizer)(hne_dense_drop1)
    hne_dense_drop2 = tf.nn.dropout(hne_dense_layer2, rate=1-keep_prob)
    
    # Combined Branch
    combined = tf.concat([ct_dense_drop2, hne_dense_drop2], axis=1)
    dense_layer3 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu, kernel_regularizer=regularizer)(combined)
    dense_drop3 = tf.nn.dropout(dense_layer3, rate=1-keep_prob)
    theta = tf.keras.layers.Dense(units=1, activation=None, use_bias=False, kernel_regularizer=regularizer)(dense_drop3)
    theta = tf.reshape(theta, [-1])
    exp_theta = tf.exp(theta) 
    
    loss = -tf.reduce_mean(tf.multiply((theta - tf.math.log(tf.reduce_sum(tf.multiply(exp_theta, R_matrix), axis=1))), tf.reshape(ystatus, [-1]))) 
    l2_loss = tf.compat.v1.losses.get_regularization_loss()
    loss = loss + l2_loss
    
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    
    saver = tf.compat.v1.train.Saver()
    
    with tf.compat.v1.Session() as sess:
        print('Graph started...')
        print('NUM_TRAIN_STEPS', NUM_TRAIN_STEPS)
        print('NUM_EPOCHES', NUM_EPOCHES)
        sess.run(tf.compat.v1.global_variables_initializer()) 
        sess.run(tf.compat.v1.local_variables_initializer())

        for ep in range(NUM_EPOCHES):
    
            batch_gen_train = get_batch_train(BATCH_SIZE, y_train, x_train_ct, x_train_hne, ystatus_train) 
            batch_gen_holdout = get_batch_holdout(BATCH_SIZE, y_holdout, x_holdout_ct, x_holdout_hne, ystatus_holdout) 
            total_loss_train = 0.0
            total_loss_holdout = 0.0
    
            for step in range(NUM_TRAIN_STEPS):
                batch_x_train_ct, batch_x_train_hne, batch_y_train, batch_ystatus_train = next(batch_gen_train) 
                batch_x_holdout_ct, batch_x_holdout_hne, batch_y_holdout, batch_ystatus_holdout = next(batch_gen_holdout)

                R_matrix_train = np.zeros([batch_y_train.shape[0], batch_y_train.shape[0]], dtype=int)
                for i in range(batch_y_train.shape[0]):
                    for j in range(batch_y_train.shape[0]):
                        R_matrix_train[i, j] = batch_y_train[j, 0] >= batch_y_train[i, 0]
                R_matrix_holdout = np.zeros([batch_y_holdout.shape[0], batch_y_holdout.shape[0]], dtype=int)
                for i in range(batch_y_holdout.shape[0]):
                    for j in range(batch_y_holdout.shape[0]):
                        R_matrix_holdout[i, j] = batch_y_holdout[j, 0] >= batch_y_holdout[i, 0]  
                          
                loss_batch_train, _ = sess.run([loss, optimizer], feed_dict={x_ct: batch_x_train_ct,
                                                                             x_hne: batch_x_train_hne,
                                                                             ystatus: batch_ystatus_train,
                                                                             R_matrix: R_matrix_train,
                                                                             keep_prob: KEEP_PROB})
                loss_batch_holdout = sess.run(loss, feed_dict={x_ct: batch_x_holdout_ct,
                                                               x_hne: batch_x_holdout_hne,
                                                               ystatus: batch_ystatus_holdout,
                                                               R_matrix: R_matrix_holdout,
                                                               keep_prob: 1})
    
                total_loss_train += loss_batch_train
                total_loss_holdout += loss_batch_holdout
    
                if (step + 1) % EVA_STEP == 0: # print loss every EVA_STEP
                    print('Average train loss at Epoch %d and Step %d is: %f' % (ep, step, total_loss_train / EVA_STEP), ';',
                          'Holdout loss is: %f' % (total_loss_holdout / EVA_STEP))
                    total_loss_train = 0.0
                    total_loss_holdout = 0.0
        save_path = saver.save(sess, checkpoint)
        print(("Model saved in file: %s" % save_path))

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    CT_FEATURE_SIZE = config['ct_feature_size']
    HNE_FEATURE_SIZE = config['hne_feature_size']
    BATCH_SIZE = config['batch_size']
    LEARNING_RATE = config['lr']
    NUM_EPOCHES = config['num_epo']
    KEEP_PROB = config['keep_prob']
    REG_SCALE = config['reg_scale']
    SELECT_SIZE = config['select_size']

    model_path = config['model_path']
    x_tr1_ct = np.loadtxt(fname=config['train_ct_feature'], delimiter=",", skiprows=1)          
    x_tr1_hne = np.loadtxt(fname=config['train_hne_feature'], delimiter=",", skiprows=1)          
    y_tr1 = np.loadtxt(fname=config['train_time'], delimiter=",", skiprows=1) 
    ystatus_tr1 = np.loadtxt(fname=config['train_status'], delimiter=",", skiprows=1) 
    x_tr2_ct = np.loadtxt(fname=config['train_ct_feature_target'], delimiter=",", skiprows=1)          
    x_tr2_hne = np.loadtxt(fname=config['train_hne_feature_target'], delimiter=",", skiprows=1)          
    y_tr2 = np.loadtxt(fname=config['train_time_target'], delimiter=",", skiprows=1) 
    ystatus_tr2 = np.loadtxt(fname=config['train_status_target'], delimiter=",", skiprows=1) 
    x_val_ct = np.loadtxt(fname=config['val_ct_feature'], delimiter=",", skiprows=1) 
    x_val_hne = np.loadtxt(fname=config['val_hne_feature'], delimiter=",", skiprows=1) 
    y_val = np.loadtxt(fname=config['val_time'], delimiter=",", skiprows=1)        
    ystatus_val = np.loadtxt(fname=config['val_status'], delimiter=",", skiprows=1) 

    NUM_TRAIN_STEPS = int((len(y_tr1) + SELECT_SIZE) / BATCH_SIZE)
    EVA_STEP = 1
    START = 1
    END = 2

    for i in range(START, END):   
        random.seed(i)
        smp_ind = random.sample(range(x_tr2_ct.shape[0]), SELECT_SIZE)
        x_tr2_smp_ct = x_tr2_ct[smp_ind,]
        x_tr2_smp_hne = x_tr2_hne[smp_ind,]
        y_tr2_smp = y_tr2[smp_ind,]
        ystatus_tr2_smp = ystatus_tr2[smp_ind,]
        
        x_train_ct = np.concatenate((x_tr1_ct, x_tr2_smp_ct), axis=0)
        x_train_hne = np.concatenate((x_tr1_hne, x_tr2_smp_hne), axis=0)
        y_train = np.concatenate((y_tr1, y_tr2_smp), axis=0)
        ystatus_train = np.concatenate((ystatus_tr1, ystatus_tr2_smp), axis=0)    
            
        print(x_tr1_ct.shape)
        print(x_tr2_smp_ct.shape)
        print(x_train_ct.shape)
        CHECKPOINT_FILE = model_path + 'combined_4layer200_dropout' + str(KEEP_PROB) + '_reg' + str(REG_SCALE) + '_batch' + str(BATCH_SIZE) + '_epo' + str(NUM_EPOCHES) + '_dup' + str(i) + '.ckpt'
        train(y_val, x_val_ct, x_val_hne, ystatus_val, y_train, x_train_ct, x_train_hne, ystatus_train, checkpoint=CHECKPOINT_FILE)