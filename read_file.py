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

import re
text = open('book.txt', 'r').read()

text = clean_text(text)
print(text)