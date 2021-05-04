# ********** Phase 1: Import Libaries  **********
import tensorflow as tf
import tensorflow_hub as hub

from official.nlp.data.squad_lib import generate_tf_record_from_json_file
from official.nlp.bert.input_pipeline import create_squad_dataset

# better optimizer for fine-tune BERT
from official.nlp import optimization

# Process for evaluation from squad to BERT
from official.nlp.data.squad_lib import convert_examples_to_features
from official.nlp.data.squad_lib import read_squad_examples
from official.nlp.data.squad_lib import write_predictions
from official.nlp.data.squad_lib import FeatureWriter

import numpy as np
import collections
import random
import json
import math
import time
import os

# ********** Phase 2: Data Preprocessing  **********

# training file, vocab file and path to output of tf files
input_meta_data = generate_tf_record_from_json_file("./data/train-v1.1.json", "./data/vocab.txt", "./data/train-v1.1.tf_record")

# save the input meta data in json format
with tf.io.gfile.GFile("./data/squad/train_meta_data", "w") as writer:
    writer.write(json.dumps(input_meta_data, indent=4) + "\n")  # \n just to make sure there is no issue with the code 

'''
* inputs are much bigger so bigger batch size can create an issue, CPU limitation
* because tweets have small sentences were as here most of time we have to work with multiple paragraph with another sentence that is question
'''
BATCH_SIZE = 4

train_dataset = create_squad_dataset(
    "./data/train-v1.1.tf_record",
    input_meta_data['max_seq_length'], # we don't have input sequence of more than 384 words
    BATCH_SIZE,
    is_training=True)  # stating it is for training set

# ********** Phase 3: Model Building  **********

# SQUAD Layer: layer to add after the BERT
# Get 2 list start and end word socre of the answer.
class BertSquadLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(BertSquadLayer, self).__init__()
        self.final_dense = tf.keras.layers.Dense(
            units=2, # 2 output units
            # Gaussian function 'TruncatedNormal' not to get high values stddev how much to spred the function
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)) 

    def call(self, inputs):
        logits = self.final_dense(inputs) # (batch_size, seq_len, 2)

        logits = tf.transpose(logits, [2, 0, 1]) # (2, batch_size, seq_len)
        unstacked_logits = tf.unstack(logits, axis=0) # split the logits [(batch_size, seq_len), (batch_size, seq_len)] 
        return unstacked_logits[0], unstacked_logits[1]

# Whole Model
class BERTSquad(tf.keras.Model):
    
    def __init__(self, name="bert_squad"):
        super(BERTSquad, self).__init__(name=name)
        
        # for full version traing change to "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4"
        self.bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
            trainable=True)
        
        self.squad_layer = BertSquadLayer()
    
    def apply_bert(self, inputs):

        # we dont use first output squad use dictionery
        _ , sequence_output = self.bert_layer([inputs["input_word_ids"], inputs["input_mask"], inputs["input_type_ids"]])
        return sequence_output

    def call(self, inputs):
        seq_output = self.apply_bert(inputs)

        start_logits, end_logits = self.squad_layer(seq_output)
        
        return start_logits, end_logits

# ********** Phase 4: Training AI  **********

TRAIN_DATA_SIZE = 88641   # total no of the training dataset 
NB_BATCHES_TRAIN = 2000   # toal no of batch we want to keep (2000*4=8000) 8000 no. of the training dataset
BATCH_SIZE = 4
NB_EPOCHS = 3
INIT_LR = 5e-5   # leraning rate for squad 5e-5
WARMUP_STEPS = int(NB_BATCHES_TRAIN * 0.1)   # warmup configuration for hyper parameter

train_dataset_light = train_dataset.take(NB_BATCHES_TRAIN)

# setting up optimize options 
optimizer = optimization.create_optimizer(
    init_lr=INIT_LR,
    num_train_steps=NB_BATCHES_TRAIN,
    num_warmup_steps=WARMUP_STEPS)

'''
* model provide 2 models start and end score it clssification problem
* if start  position is at the position 50 and model have high posobility at position 50 then we have low loss.
* find the loss of both position and get the mean this way we get the total loss.
'''
def squad_loss_fn(labels, model_outputs):
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    start_logits, end_logits = model_outputs

    # get the probability
    start_loss = tf.keras.backend.sparse_categorical_crossentropy(
        start_positions, start_logits, from_logits=True)
    end_loss = tf.keras.backend.sparse_categorical_crossentropy(
        end_positions, end_logits, from_logits=True)
    
    total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2

    return total_loss

train_loss = tf.keras.metrics.Mean(name="train_loss")

next(iter(train_dataset_light))

# compiling the BERT Model 
bert_squad.compile(optimizer, squad_loss_fn)

# save the trained model 
checkpoint_path = "./ckpt_bert_squad/"

ckpt = tf.train.Checkpoint(bert_squad=bert_squad)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")


## -- Custom training
for epoch in range(NB_EPOCHS):
    print("Start of epoch {}".format(epoch+1))
    start = time.time()
    
    train_loss.reset_states()
    
    for (batch, (inputs, targets)) in enumerate(train_dataset_light):
        # help us know what happens in the weights, this information of compute help us optimze the Gradian and apply the optimizer
        with tf.GradientTape() as tape:
            model_outputs = bert_squad(inputs)
            loss = squad_loss_fn(targets, model_outputs)
        
        gradients = tape.gradient(loss, bert_squad.trainable_variables)
        # modify the weight of the layers to decrese the loss
        optimizer.apply_gradients(zip(gradients, bert_squad.trainable_variables))
        
        train_loss(loss)
        
        if batch % 50 == 0:
            print("Epoch {} Batch {} Loss {:.4f}".format(
                epoch+1, batch, train_loss.result()))
        
        # just a point to to save the model
        if batch % 500 == 0:
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint for epoch {} at {}".format(epoch+1, ckpt_save_path))
    print("Time taken for 1 epoch: {} secs\n".format(time.time() - start))