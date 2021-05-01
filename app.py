from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# load the model from disk

app = Flask(__name__)
summarizer = pipeline("summarization")

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		body = request.form['message']
		print(body)
		sum_len = int(len(body)*0.2)
		# use bart in pytorch
		
		txt_summary = summarizer(body, min_length=25, max_length=sum_len)
		txt_summary = txt_summary[0]['summary_text']
	return render_template('result.html',summary = txt_summary)


if __name__ == '__main__':
	app.run(debug=True)