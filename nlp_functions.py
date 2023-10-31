import re
import nltk
nltk.data.path.append("./nltk_data")
from nltk.tokenize import sent_tokenize
from mailparser_reply import EmailReplyParser #https://github.com/alfonsrv/mail-parser-reply

def process_email_body(input_data):
    # If the input is a single string, return result from remove_signature function
    if isinstance(input_data, str):
        return remove_signature_from_email(input_data)
    
    # If the input is a list of strings, process each string and store the results in a new list
    elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
        result = []
        for item in input_data:
            result.append(remove_signature_from_email(item))
        return " ".join(result)
    
    # If the input is neither a string nor a list of strings, raise a ValueError
    else:
        raise ValueError("Input must be either a string or a list of strings.")
    
def remove_signature_from_email(emailBody):
    languages = ['en', 'de']
    clean_emails = EmailReplyParser(languages=languages).read(text=emailBody).replies # returns a list of EmailReplys
    clean_email = " ".join(obj.body for obj in clean_emails) #concatenate to string
    return clean_email

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

# https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
# split_into_sentences is an alternative to nltk's sent_tokenize
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'
def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences
