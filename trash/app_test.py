from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import summary as sc

from transformers import pipeline

# load the model from disk

app = Flask(__name__)
summarizer = pipeline("summarization")

@app.route('/', methods=["GET"])
def home():
	if request.method == "GET":
		return render_template("index.html")

@app.route('/context')
def context():
	return render_template('context.html')

@app.route('/url_link')
def url_link():
	return render_template('url_link.html')

@app.route('/upload')
def upload():
	return render_template('upload.html')

@app.route('/qa')
def qa():
	return render_template('qa.html')

@app.route('/context_summary',methods=['POST'])
def context_summary():

	if request.method == 'POST':
		text = request.form['message']
		summary = sc.summary_result(text, summarizer)

	return render_template('context.html',summary = txt_summary)

if __name__ == '__main__':
	app.run(debug=True)