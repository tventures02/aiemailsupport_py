import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def contains_question(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Check if any sentence ends with a question mark
    for sentence in sentences:
        if sentence.strip().endswith('?'):
            return True
    return False


def find_matches(input_text, substrings):
    matches = [substring for substring in substrings if substring in input_text]
    return ', '.join(matches)