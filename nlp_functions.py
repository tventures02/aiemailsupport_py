import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

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

def remove_sentences(paragraph, matches):
    # Split the paragraph into sentences
    sentences = re.split('(?<=[.!?])\s+', paragraph)
    
    # convert string with comma delimited phrases to array of strings with no trailing white space
    matches = [item.strip() for item in matches.split(',')]
    
    # Remove sentences that contain any of the matches
    filtered_sentences = [sentence for sentence in sentences if not any(match in sentence for match in matches)]

    # Join the filtered sentences back into a paragraph
    return ' '.join(filtered_sentences)

def count_tokens(documents):
    return sum([len(word_tokenize(document.text)) for document in documents])

