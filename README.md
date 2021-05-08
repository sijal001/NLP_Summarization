<h1 align="center"> <strong>Text Summarization</strong> </h1>


![Project Image](https://image.freepik.com/free-photo/black-male-student-underlining-important-information-textbook-using-pencil-while-making-history-research-university-canteen-during-lunch-phone-coffee-food-resting-table_273609-7535.jpg)

> <p> <strong> The digital era also means that customers want to get answers not from one book but from thousands immediately. </strong> </p>

---

## **Table of Contents**
You're sections headers will be used to reference location of destination.

- [Description](#description)
- [How To Use](#how-to-use)
- [Repo Artitecture](#repo-artitecture)
- [Next Step](#next-step)
- [License](#license)
- [Author Info](#author-info)

---

## **Description**

<p align="justify">
An online dashboard where the user can select or upload an e-book and the summary is automatically generated. The contents are summarizing by paragraph. The users can also enter paragraphs of information to be summarized. The users can query questions to be answered according to the content of the book. The queries are answered according to the content of multiple books on the same or similar topics.  
</p>
<p align="justify">
Ttokenization, stemming and lemmatization are used for exploration of text-based datasets. As for text ummarization Explore state-of-the-art algorithms has been used.
Transformer model has been used and pre-trained model perfomance has been evaluated. Development and deployment of the dashboard are done in Heroku.
</p>

<p align="justify">
I am using the Bert_Large_CNN, the transformer pipline to do the summarization. User can use the webpage to easyly upload their file, use url or use context to get the summarization.
User can also use the QA service. User can paste or write the context they want and ask system a question. Then system try it best to provide with best result answer possible.
</p>

<br/>

## **Technologies**
<br/>

| Library       | Used to                                        |
| ------------- | :----------------------------------------------|
| Flask         | to scale up to complex applications.           |
| gunicorn      | a Python WSGI HTTP Server for UNIX.            |
| itsdangerous  | to ensure that a token has not been tampered.  |
| Jinja2        | a combination of Django templates and python.  |
| MarkupSafe    | to mitigates injection attacks                 |
| Werkzeug      | to build all sorts of end user applications    |
| numpy         | to scientific uses                             |
| scipy         | for fast N-dimensional array manipulation      |
| scikit-learn  | for machine learning built on top of SciPy     |
| matplotlib    | for creating visualizations                    |
| pandas        | to work with data structure and manupulate it  |
| nltk          | for natural language processing.               |
| regex         | to manupulate with the text and strings value  |
| transformers  | to perform multiple tasks on texts             |
| beautifulsoup4| to scrape information from web pages.          |

[**↥ Back To The Top**](#table-of-contents)

---

## **How To Use**

### **Installation** 

`Python Ver. '3.8'`

**Note:** Just use command below to install the required libary with correct version to run the program smoothly.

`pip install -r requiement.txt`


1. Inside Bert_QA directory create 2 new drictory and name it `ckpt_bert_squad` and `data`.
2. Download import file and move it to data directory.
    * [training set](https://drive.google.com/file/d/1zwSjQX2gNb2EYldVoR5sCibnSDz9DQtH/view?usp=sharing)
    * [dev set](https://drive.google.com/file/d/1YjCrVa3906b4KCWTQu3KySY7gE1CPqpC/view?usp=sharing)
    * [vocab.txt](https://drive.google.com/file/d/1kp8ApuoHSjROy0Rca0BNQ6YrnbpRtYCk/view?usp=sharing)
    * [evaluation script](https://drive.google.com/file/d/1DKhqdc8tdMnZ4EzLtW2zuG0Pf6z6H3vF/view?usp=sharing )

3. Inside Bert_QA directory run `BERT_QA_model.py` script file inside dir. (This generate traing file)
4. Run the `app.py` file to host the application locally.


[**↥ Back To The Top**](#table-of-contents)

---

## **Repo Artitecture**
```
NLP_Summarization
│
│   README.md               :explains the project
│   requirements.txt        :packages to install to run the program
│   .gitignore              :specifies which files to ignore when pushing to the repository
│__   
│   static                  :directory contain all the main css file to style html.
│   │
│   │ css.css               :css file to style home page
│   │ style.css             :seconday css file for other page (optional)
│__   
│   templates               :directory contain all the main html that work as a dashboard.
│   │
│   │ Qa.html               :page for question answering service
│   │ context.html          :page for user to paste their context and get their summarzation.
│   │ index.html            :home page for website.
│   │ upload.html           :page for user to upload their files and get their summarzation.
│   │ url.html              :page for user to paste their url and get their summarzation.
│__   
│   Bert_QA                 :directory contain all the main .py files related to Question answeing model.
│   │
│   │ BERT_QA_Single_prediction.py   :python script used for predicting the answer from the paragraph.
│   │ BERT_QA_evaluation.py :python script used for evaluating the models predition results.
│   │ BERT_QA_model.py      :python script used for create and saving the train model.
│   │ __init__.py           :python script used for initial file whole importing to other directory
│   │ evaluate-v1.1.py	    :python script used for evaluating the models predition results.
│   │
│	 summary.py             :python script file that summarize from context, url link or files.
│	 app.py                 :python script file to deploy model and html files for web application.
│	 Procfile               :files used for deployment.
```

[**↥ Back To The Top**](#table-of-contents)

---

## **Next Step**

- Addition feature: **Text-to-Speech**.
- Addition summary option: **Abstract** for Books and **Executive** for Report.
- **E-Mail copy of sumarization** as txt file to user.
- Option to **save audio or text file**.
- **Read other extention** files as pdf, epud, etc.
- Improvement of model (model options as Pegasus).
- Training QA with BERT_Large, SQuAD Ver. 2 and full dataset.

[**↥ Back To The Top**](#table-of-contents)

---
## **License**

Copyright (c) [2021] [Sijal Kumar Joshi]

<p align="justify">
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
</p>
<p align="justify">
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
</p>
<p align="justify">
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
</p>

[**↥ Back To The Top**](#table-of-contents)

---

## **Author Info**

- Linkedin - [Sijal Kumar Joshi](https://twitter.com/jamesqquick)
- Github   - [sijal001](https://github.com/sijal001)

[**↥ Back To The Top**](#table-of-contents)