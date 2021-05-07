from Bert_QA import BERT_QA_Single_prediction as bqa
from official.nlp.bert.tokenization import FullTokenizer
import tensorflow as tf
import tensorflow_hub as hub

import warnings
warnings.filterwarnings("ignore")

bert_squad = bqa.BERTSquad()
print(' * * * * * * * * * * import correctly * * * * * * * * * * ')


# save the trained model 
checkpoint_path = "./Bert_QA/ckpt_bert_squad/"
ckpt = tf.train.Checkpoint(bert_squad=bert_squad)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
ckpt.restore(ckpt_manager.latest_checkpoint)
print("Latest checkpoint restored!!")

# tokenize new words
my_bert_layer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
    trainable=False)
vocab_file = my_bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = my_bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

my_context = '''Neoclassical economics views inequalities in the distribution of income as arising from differences in value added by labor, capital and land. Within labor income distribution is due to differences in value added by different classifications of workers. In this perspective, wages and profits are determined by the marginal value added of each economic actor (worker, capitalist/business owner, landlord). Thus, in a market economy, inequality is a reflection of the productivity gap between highly-paid professions and lower-paid professions.'''
my_question = '''What are examples of economic actors?'''

predicted_answer = bqa.predict_answer(bert_squad, my_context, my_question, tokenizer)

print("**********")
print(predicted_answer)
print("**********")

    """
my_bert_layer = hub.KerasLayer(
"https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
trainable=False)
vocab_file = my_bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = my_bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case

my_context = '''Neoclassical economics views inequalities in the distribution of income as arising from differences in value added by labor, capital and land. Within labor income distribution is due to differences in value added by different classifications of workers. In this perspective, wages and profits are determined by the marginal value added of each economic actor (worker, capitalist/business owner, landlord). Thus, in a market economy, inequality is a reflection of the productivity gap between highly-paid professions and lower-paid professions.'''

#my_question = '''What philosophy of thought addresses wealth inequality?'''
my_question = '''What are examples of economic actors?'''
#my_question = '''In a market economy, what is inequality a reflection of?'''


    my_input_dict, my_context_words, context_tok_to_word_id, question_tok_len = create_input_dict(my_question, my_context)
    start_logits, end_logits = bert_squad(my_input_dict, training=False)
    start_logits_context = start_logits.numpy()[0, question_tok_len+1:]
    end_logits_context = end_logits.numpy()[0, question_tok_len+1:]
    start_word_id = context_tok_to_word_id[np.argmax(start_logits_context)]
    end_word_id = context_tok_to_word_id[np.argmax(end_logits_context)]
    pair_scores = np.ones((len(start_logits_context), len(end_logits_context)))*(-1E10)
    for i in range(len(start_logits_context-1)):
        for j in range(i, len(end_logits_context)):
            pair_scores[i, j] = start_logits_context[i] + end_logits_context[j]
    pair_scores_argmax = np.argmax(pair_scores)
    start_word_id = context_tok_to_word_id[pair_scores_argmax // len(start_logits_context)]
    end_word_id = context_tok_to_word_id[pair_scores_argmax % len(end_logits_context)]
    predicted_answer = ' '.join(my_context_words[start_word_id:end_word_id+1])
    """