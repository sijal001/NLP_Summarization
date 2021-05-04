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

# ********** Phase 2: Load Models **********
# save the trained model 
checkpoint_path = "./ckpt_bert_squad/"

ckpt = tf.train.Checkpoint(bert_squad=bert_squad)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")
else:
    print("No Model save train the model")

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

# ********** Stage 5: Evaluation **********
'''
* This code is specific to SQuAD. for evaluation inputs 
* Googel wrote hundreds of line in order to have somethiong that is optimize to get the correct file for the evaluation script (we use google function) 
'''

## -- Prepare evaluation

# Get the dev set in the session
eval_examples = read_squad_examples("dev-v1.1.json", is_training=False, version_2_with_negative=False)

# Define the function that will write the tf_record file for the dev set
eval_writer = FeatureWriter(filename=os.path.join("./data/squad/", "eval.tf_record"),  is_training=False)

# Create a tokenizer for future information needs
# for full version traing change to "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4"
my_bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
vocab_file = my_bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = my_bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

# Define the function that add the features (feature is a protocol in tensorflow) to our eval_features list
def _append_feature(feature, is_padding):
    if not is_padding:
        eval_features.append(feature)
    eval_writer.process_feature(feature)

# Create the eval features and the writes the tf.record file
eval_features = []
dataset_size = convert_examples_to_features(
    examples=eval_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    output_fn=_append_feature,
    batch_size=4)

eval_writer.close()

# Load the ready-to-be-used dataset to our session
BATCH_SIZE = 4

eval_dataset = create_squad_dataset(
    "./data/squad/eval.tf_record",
    384,        #input_meta_data['max_seq_length'],
    BATCH_SIZE,
    is_training=False)  # its not for trainig

## -- Making the predictions
'''
Need to make correct input format according to google
'''

# Defines a certain type of collection (like a dictionary).nametupale create tuble with element with name attached to it.
# kind of dictionary but tuple 
RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

# Returns each element of batched output once at a time
def get_raw_results(predictions):
    for unique_ids, start_logits, end_logits in zip(predictions['unique_ids'],
                                                    predictions['start_logits'],
                                                    predictions['end_logits']):
        # yeild to return on after the other.
        yield RawResult(
            unique_id=unique_ids.numpy(),
            start_logits=start_logits.numpy().tolist(), # format to use for the google function to write our files
            end_logits=end_logits.numpy().tolist())

# Making our predictions!
all_results = []
for count, inputs in enumerate(eval_dataset):
    x, _ = inputs       # just get the input not anwsers
    unique_ids = x.pop("unique_ids")  # remove the ids from x and saving to unique_ids
    start_logits, end_logits = bert_squad(x, training=False)
    output_dict = dict(
        unique_ids=unique_ids,
        start_logits=start_logits,
        end_logits=end_logits)
    for result in get_raw_results(output_dict):
        all_results.append(result)
    
    # we have long batche "2709" so wee keep infor
    if count % 100 == 0:
        print("{}/{}".format(count, 2709))

# Write the predictions in a json file that will work with the evaluation script
output_prediction_file = "./data/squad/predictions.json"
output_nbest_file = "./data/squad/nbest_predictions.json"
output_null_log_odds_file = "./data/squad/null_odds.json"

write_predictions(
    eval_examples,
    eval_features,
    all_results,
    20,
    30,
    True,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose=False)


# Making single prediction

my_context = '''Neoclassical economics views inequalities in the distribution of income as arising from differences in value added by labor, capital and land. Within labor income distribution is due to differences in value added by different classifications of workers. In this perspective, wages and profits are determined by the marginal value added of each economic actor (worker, capitalist/business owner, landlord). Thus, in a market economy, inequality is a reflection of the productivity gap between highly-paid professions and lower-paid professions.'''
my_question = '''What are examples of economic actors?'''

def single_prediction(question, context):
    
    start_logits, end_logits = bert_squad(question, training=False)
    output_dict = dict(
        unique_ids=unique_ids,
        start_logits=start_logits,
        end_logits=end_logits)
    
    print(get_raw_results(output_dict))