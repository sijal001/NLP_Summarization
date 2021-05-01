import re
import bs4 as bs
import nltk
nltk.download('punkt')
import urllib.request
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

def url_text(url_link):
    source = urllib.request.urlopen(url_link)

    soup = bs.BeautifulSoup(source, 'lxml')

    text = ""
    for paragraph in soup.find_all('p'):
        text += paragraph.text

    return text


# if txt file do the cleaning here
def clean_text(text):
    
    # remove reference citation
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    # Removing the @
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    # Removing the URL links
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    # Keeping only letters and numbers
    text = re.sub(r"[^a-zA-Z0-9.!?']", ' ', text)
    # Removing additional whitespaces
    text = re.sub(r" +", ' ', text)

    return text

def paragraph_list(sentences, sentence_limit=6, type="book"):
    sentence_limit = sentence_limit
    sentence_batch_lst = []
    sentence_batch = []

    # If book drop main two sentences that mostly contain acknowledgement and others
    if type=="book":
        for sentence in sentences[2:]:
            if len(sentence_batch)==sentence_limit:
                sentence_batch_lst.append(' '.join(sentence_batch))
                sentence_batch = []
            else:
                sentence_batch.append(sentence)
    else:
        for sentence in sentences:
        
            if len(sentence_batch)==sentence_limit:
                sentence_batch_lst.append(' '.join(sentence_batch))
                sentence_batch = []
            else:
                sentence_batch.append(sentence)

    if len(sentence_batch) > 0:
            sentence_batch_lst.append(' '.join(sentence_batch))
            sentence_batch = []
    
    return sentence_batch_lst


def summary_output():
    summary_result = []

    for paragraph in paragraphs:
        body = paragraph

        # use bart in pytorch
        summarizer = pipeline("summarization")
        summary = summarizer(body, min_length=5)
        summary_result.append(summary[0]['summary_text'])

    summary_result = ' '.join(summary_result)
    return summary_result

# url_link = input("Link: ")
url_link = "https://www.gutenberg.org/cache/epub/103/pg103.txt"

text = url(url_link)
text = clean_text(text)
sentences = nltk.sent_tokenize(text)
paragraphs = paragraph_list(sentences)
summary = summary_output()