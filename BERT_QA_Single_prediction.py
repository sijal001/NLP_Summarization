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
        

# ********** Phase 3: Load Models **********
my_bert_layer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
    trainable=False)
vocab_file = my_bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = my_bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)


def is_whitespace(c):
    '''
    Tell if a chain of characters corresponds to a whitespace or not.
    '''
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def whitespace_split(text):
    '''
    Take a text and return a list of "words" by splitting it according to
    whitespaces.
    '''
    doc_tokens = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens

def tokenize_context(text_words):
    '''
    Take a list of words (returned by whitespace_split()) and tokenize each word
    one by one. Also keep track, for each new token, of its original word in the
    text_words parameter.
    '''
    text_tok = []
    tok_to_word_id = []
    for word_id, word in enumerate(text_words):
        word_tok = tokenizer.tokenize(word)
        text_tok += word_tok
        tok_to_word_id += [word_id]*len(word_tok)
    return text_tok, tok_to_word_id

def tokenize_context(text_words):
    '''
    Take a list of words (returned by whitespace_split()) and tokenize each word
    one by one. Also keep track, for each new token, of its original word in the
    text_words parameter.
    '''
    text_tok = []
    tok_to_word_id = []
    for word_id, word in enumerate(text_words):
        word_tok = tokenizer.tokenize(word)
        text_tok += word_tok
        tok_to_word_id += [word_id]*len(word_tok)
    return text_tok, tok_to_word_id

def tokenize_context(text_words):
    '''
    Take a list of words (returned by whitespace_split()) and tokenize each word
    one by one. Also keep track, for each new token, of its original word in the
    text_words parameter.
    '''
    text_tok = []
    tok_to_word_id = []
    for word_id, word in enumerate(text_words):
        word_tok = tokenizer.tokenize(word)
        text_tok += word_tok
        tok_to_word_id += [word_id]*len(word_tok)
    return text_tok, tok_to_word_id

def create_input_dict(question, context):
    '''
    Take a question and a context as strings and return a dictionary with the 3
    elements needed for the model. Also return the context_words, the
    context_tok to context_word ids correspondance and the length of
    question_tok that we will need later.
    '''
    question_tok = tokenizer.tokenize(my_question)

    context_words = whitespace_split(context)
    context_tok, context_tok_to_word_id = tokenize_context(context_words)

    input_tok = question_tok + ["[SEP]"] + context_tok + ["[SEP]"]
    input_tok += ["[PAD]"]*(384-len(input_tok)) # in our case the model has been
                                                # trained to have inputs of length max 384
    input_dict = {}
    input_dict["input_word_ids"] = tf.expand_dims(tf.cast(get_ids(input_tok), tf.int32), 0)
    input_dict["input_mask"] = tf.expand_dims(tf.cast(get_mask(input_tok), tf.int32), 0)
    input_dict["input_type_ids"] = tf.expand_dims(tf.cast(get_segments(input_tok), tf.int32), 0)

    return input_dict, context_words, context_tok_to_word_id, len(question_tok)


# ********** PREDICTION **********
my_context = '''Neoclassical economics views inequalities in the distribution of income as arising from differences in value added by labor, capital and land. Within labor income distribution is due to differences in value added by different classifications of workers. In this perspective, wages and profits are determined by the marginal value added of each economic actor (worker, capitalist/business owner, landlord). Thus, in a market economy, inequality is a reflection of the productivity gap between highly-paid professions and lower-paid professions.'''
my_question = '''What are examples of economic actors?'''

my_input_dict, my_context_words, context_tok_to_word_id, question_tok_len = create_input_dict(my_question, my_context)
start_logits, end_logits = bert_squad(my_input_dict, training=False)

# We remove the ids corresponding to the question and the ["SEP"] token:
start_logits_context = start_logits.numpy()[0, question_tok_len+1:]
end_logits_context = end_logits.numpy()[0, question_tok_len+1:]

# First easy interpretation:
start_word_id = context_tok_to_word_id[np.argmax(start_logits_context)]
end_word_id = context_tok_to_word_id[np.argmax(end_logits_context)]

# "Advanced" - making sure that the start of the answer is before the end
pair_scores = np.ones((len(start_logits_context), len(end_logits_context)))*(-1E10)
for i in range(len(start_logits_context-1)):
    for j in range(i, len(end_logits_context)):
        pair_scores[i, j] = start_logits_context[i] + end_logits_context[j]
pair_scores_argmax = np.argmax(pair_scores)

start_word_id = context_tok_to_word_id[pair_scores_argmax // len(start_logits_context)]
end_word_id = context_tok_to_word_id[pair_scores_argmax % len(end_logits_context)]

# geting the answer
predicted_answer = ' '.join(my_context_words[start_word_id:end_word_id+1])
