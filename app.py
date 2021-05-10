from flask import Flask,render_template,url_for,request
import pandas as pd 
import summary as sc
import numpy as np
from transformers import pipeline
from Bert_QA import BERT_QA_Single_prediction as bqa
from official.nlp.bert.tokenization import FullTokenizer
import tensorflow as tf
import tensorflow_hub as hub

# load the model from disk

app = Flask(__name__)
summarizer = pipeline("summarization")
bert_squad = bqa.BERTSquad()

# save the trained model 
checkpoint_path = "./Bert_QA/ckpt_bert_squad/"
ckpt = tf.train.Checkpoint(bert_squad=bert_squad)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
ckpt.restore(ckpt_manager.latest_checkpoint)
print("Latest checkpoint restored!!")


@app.route('/')
def index():
	return render_template("index.html")

@app.route('/index.html')
def home():
	return render_template("index.html")

@app.route('/context.html', methods=["GET", "POST"])
def context():
	if request.method == 'GET':
		return render_template('context.html')

	elif request.method == 'POST':
		text = request.form['user_context']
		text_summary = sc.summary_result(text, summarizer)
		return render_template('context.html',summary=text_summary, user_text=text)

@app.route('/url_link.html', methods=["GET", "POST"])
def url_link():
	if request.method == 'GET':
		return render_template('url_link.html', link="Paste your link here...")

	elif request.method == 'POST':
		url_link = request.form['user_context']
		text = sc.url_text(url_link)
		text_summary = sc.summary_result(text, summarizer)
		return render_template('url_link.html',summary=text_summary, link=url_link)

@app.route('/upload.html', methods=["GET", "POST"])
def upload():
	if request.method == 'GET':
		return render_template('upload.html')
	
	elif request.method == 'POST':
		upload = request.form['user_context']
		text = sc.file_text(upload)
		text_summary = sc.summary_result(text, summarizer)
		return render_template('upload.html',summary=text_summary)

@app.route('/QA.html', methods=["GET", "POST"])
def qa():
	if request.method == 'GET':
		return render_template('QA.html', question="What is your question?")
	
	elif request.method == 'POST':
		my_context = request.form['my_context']
		my_question = request.form['my_question']

		my_bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
		vocab_file = my_bert_layer.resolved_object.vocab_file.asset_path.numpy()
		do_lower_case = my_bert_layer.resolved_object.do_lower_case.numpy()
		tokenizer = FullTokenizer(vocab_file, do_lower_case)

		my_input_dict, my_context_words, context_tok_to_word_id, question_tok_len = bqa.create_input_dict(my_question, my_context, tokenizer)
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

		return render_template('QA.html',answer=predicted_answer, question=my_question, user_context=my_context)

if __name__ == '__main__':
	app.run(debug=True)