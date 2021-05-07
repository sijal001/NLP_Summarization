"""
[Developer]: [Sijal Kumar Joshi]

[Text Summarization]

Returns:
    [string]: [summary for the content provided by the user]
"""

# import important libaries
import re
import bs4 as bs
import nltk
nltk.download('punkt')
import urllib.request
from transformers import AutoModelForTokenClassification, AutoTokenizer

'''
from transformers import pipeline
summarizer = pipeline('summarization')
'''

# grab paragraph from the link
def url_text(url_link):
    source = urllib.request.urlopen(url_link)
    soup = bs.BeautifulSoup(source, 'lxml')
    text = ""
    print("Collecting Paragraph.")
    # get the whole paragraph of the website and make it as a string
    for paragraph in soup.find_all('p'):
        text += paragraph.text

    print("Paragraph collected.")
    return text

# read file and return text files
def file_text(link: str):
    # open the text file and provide the containt
    text = open(link, 'r').read()
    return text

# Clean the text contents
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

# make the paragraph of tall the sentences 
def paragraph_list(sentences, sentence_limit=6, word_limit=1000):
    sentence_batch_lst = []
    sentence_batch = []

    for sentence in sentences:
        # This skip any sentences with very long sentence.
        if len(sentence) > word_limit:
            continue
        
        #  when list hits the amount of sentences for paragram create a pragram and empy list
        if len(sentence_batch)==sentence_limit:
            sentence_batch_lst.append(' '.join(sentence_batch))
            sentence_batch = []
        # if amount of sentence is not enough append to the list to create paragraph
        else:
            sentence_batch.append(sentence)

    # if at the very end there is unappended list beacuse not enough sentence append it
    if len(sentence_batch) > 0:
            sentence_batch_lst.append(' '.join(sentence_batch))
            sentence_batch = []
    
    return sentence_batch_lst

# create the summay of the content given
def summary_output(paragraphs, summarizer):
    summary_result = []
    

    # content contains multiple paragraph so summarize every contain, make its list 
    for paragraph in paragraphs:
        
        # use bart in pytorch
        summary = summarizer(paragraph, min_length=5)
        summary_result.append(summary[0]['summary_text'])

    # join the list of every summary and provide one full summary
    summary_result = ' '.join(summary_result)
    
    return summary_result

# function to provide the result 
def summary_result(text, summarizer):
    text = clean_text(text)
    sentences = nltk.sent_tokenize(text)
    paragraphs = paragraph_list(sentences)
    
    return summary_output(paragraphs, summarizer)

"""
# testing Script
book_link = "https://www.gutenberg.org/cache/epub/103/pg103.txt"
artical_link = "https://www.nytimes.com/2021/05/01/us/susan-wright-sixth-district-texas.html"

text = url_text(artical_link)

summary = summary_result(text, summarizer)

print(summary)   
"""