import spacy
import stanza
import re
from spacy.tokens import Doc

# SOURCE: https://stackoverflow.com/questions/65160277/spacy-tokenizer-with-only-whitespace-rule
# by user "Sofie VL"
# START 
class WhitespaceTokenizer(object):
    """Tokenizer splitting text on whitespaces only, for processing of texts
    with the SpaCy language model. With the default SpaCy tokenizer, the length
    of the doc was bigger than the length of the list with the gold labels.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

# loading the spaCy and Stanza language models for English and Spanish
spacy_en = spacy.load("en_core_web_md")
spacy_en.tokenizer = WhitespaceTokenizer(spacy_en.vocab)
spacy_es = spacy.load("es_core_news_md")
spacy_es.tokenizer = WhitespaceTokenizer(spacy_es.vocab)
# END

stanza_en = stanza.Pipeline("en", processors="tokenize,ner", 
                            package={"ner": ["conll03"]},
                            tokenize_pretokenized=True)
stanza_es = stanza.Pipeline("es", processors="tokenize,ner", 
                            package={"ner": ["conll02"]},
                            tokenize_pretokenized=True)

def load_europarl(filepath):
    """Load the data from a europarl conll02-file

    args: filepath (string, full path of the europarl file)

    return: words (list of all words in the file), labels (list of all labels),
    text (string of continuous text)

    note: the file path depends on the storage location of the file on the 
    computer, and can vary from computer to computer
    """
    
    words = []
    labels = []

    with open(filepath, "r", encoding="utf-8") as infile:
        for line in infile:
            parts = line.split("\t")

            if len(parts) > 1:
                label = parts[1]

                # remove new line
                label = label[:-1]

                words.append(parts[0])
                labels.append(label)

    text = " ".join(words)

    return words, labels, text


def load_subtitles(filepath):
    """Load movie subtitle txt-file, and remove blank lines and line breaks

    args: filepath (string, full path of the subtitle file)

    return: words (list of all words in the file), text (string of continuous 
    text without blank lines and line breaks)

    note: the file path depends on the storage location of the file on the 
    computer, and can vary from computer to computer
    """

    text = ""

    with open(filepath, "r", encoding="latin-1") as infile:
        for line in infile:

            # if the line is not empty add it to the text
            if line.strip():
                text += line.strip("\n") + " "

    # add whitespaces before/after punctuation marks to facilitate tokenization
    text = re.sub(r'([.,:!?)])', r' \1', text)
    text = re.sub(r'([¿¡(])', r'\1 ', text)
    text = re.sub(r'\'', r' \' ', text)
    text = re.sub(r'\\', r'', text)

    words = text.split()

    return words, text

def ner(text, model, lang):
    """Process the given text, and return the list of predicted labels

    args: text (string, continuous text), model (string, language model to be 
    used i.e. spaCy or Stanza), lang (string, language of the text)

    return: list of all predicted labels (including recognized Named Entities
    as well as words which are not Named Entities) in the BIO(ES) format

    note: when specifying the language, please use "en" for English and "es" 
    for Spanish, and please write the names of the language models in lower 
    case letters only
    """

    if model == "spacy":
        if lang == "en":
            doc = spacy_en(text)
            # ent_iob_: return the Named Entities in the BIO format, and the
            # non-entities as well ("O")
            # ent_type_: type of the entity according to the SpaCy tag set
            preds = [doc[i].ent_iob_ + "-" + doc[i].ent_type_ for i in range(len(doc))]
        elif lang == "es":
            doc = spacy_es(text)
            preds = [doc[i].ent_iob_ + "-" + doc[i].ent_type_ for i in range(len(doc))]
    elif model == "stanza":
        if lang == "en":
            doc = stanza_en(text)
            # token.ner: return the Named Entity tag of the current token
            preds = [token.ner for sent in doc.sentences for token in sent.tokens]
        elif lang == "es":
            doc = stanza_es(text)
            preds = [token.ner for sent in doc.sentences for token in sent.tokens]
    return preds


def postprocess_labels(pred_labels):
    """Transform the fine-grained labels predicted by SpaCy into the 4 label
    format (PER, LOC, ORG, MISC), in which the europarl-data is annotated
    
    args: pred_labels (list of all predicted labels, fine-grained)
    
    return: preprocessed (list of all labels, transformed into 4 label format)
    
    note: the transformation is done according to our findings on the europarl-
    data: PER = PERSON; LOC = GPE, LOC; ORG = ORG; MISC = NORP, LANGUAGE, EVENT,
    (ORG), LAW; O = O, FAC, PRODUCT, WORK_OF_ART, DATE, TIME, PERCENT, MONEY,
    QUANTITY, ORDINAL, CARDINAL"""

    # replacement_label as a dictonary.
    replacement_label = {"PERSON": "PER", "GPE": "LOC"}

    # updating the dictionary. The labels in the list are those that need to be 
    # replaced. Whilst the second argument is the label they're being replaced 
    # with.
    replacement_label.update(dict.fromkeys(['NORP', 'LANGUAGE', 'EVENT', 'LAW'], 
                                           'MISC'))
    replacement_label.update(dict.fromkeys(['FAC', 'PRODUCT', 'WORK_OF_ART', 
                                            'DATE', 'TIME', 'PERCENT', 'MONEY', 
                                            'QUANTITY', 'ORDINAL', 'CARDINAL'], 
                                            'O'))
    
    # re.sub() from the "re" (regular expresion) modul. Replaces ALL instances 
    # of the FIRST argument with the SECOND argument that are in the THIRD 
    # argument
    # '\b' is needed as it indicated the begining and end of a word, so that we
    # only replace labels that are exactly like the given argument and not 
    # words that contain the label. 
    for old, new in replacement_label.items():
        pred_labels = [re.sub(r'\b{}\b'.format(old), new, label) for label in pred_labels]

    # replaces the hanging '-' and hanging iob's from the labels.
    # For loop through each label, checked if a hanging '-' or iob. O in the 
    # beginning means '-' at the end means iob. Then replace them with O
    # One could also put each iob + label varient in the dic. though that get's
    # quite big; 'O' only appears if it's a non-entity. 
    postprocessed = ['O' if label.startswith('O-') or label.endswith('-O') else label for label in pred_labels]

    return postprocessed


def label_match(label1, label2):
    """Check if two labels are matching
    
    args: label1, label2 (strings)
    
    return: a truth value"""

    if ((label1 == "O" and label2 == "O") or 
        ("PER" in label1 and "PER" in label2) or 
        ("LOC" in label1 and "LOC" in label2) or 
        ("ORG" in label1 and "ORG" in label2) or 
        ("MISC" in label1 and "MISC" in label2)):
        return True
    else:
        return False


def eval_europarl(word_list, gold_labels, pred_labels, model):
    """Evaluate the europarl-file: compare predicted labels and the gold 
    labels, i.e. give the accuracy and return a list of all words which were
    annotated with a different label as their gold label

    args: word_list (list of all words in the europarl-file), gold_labels
    (list of all gold labels), pred_labels (list of all predicted labels 
    [including recognized Named Entities as well as words which are not Named 
    Entities] in the BIOES format), model (string, language model to be used 
    i.e. spaCy or Stanza)

    return: accuracy (float, accuracy of pred_labels with respect to the
    gold_labels), differences (list of lists, consisting of word, gold label
    and predicted label)

    note: please write the names of the language models in lower case letters 
    only
    """

    if len(gold_labels) == len(pred_labels):
        accuracy = 0.0
        diff_indexes = []
        differences = []
        
        if model == "spacy":
            pred_labels = postprocess_labels(pred_labels)

        for i in range(len(gold_labels)):
            # if the current label matches with the corresponding gold label
            # add 1 to the accuracy counter else add the current index to the
            # list of indexes of words which differ from the gold labels
            if label_match(gold_labels[i], pred_labels[i]):
                accuracy += 1.0
            else:
                diff_indexes.append(i)

        # create the list differences
        for index in diff_indexes:
            diff = [index, word_list[index], gold_labels[index], pred_labels[index]]
            differences.append(diff)

        # divide the accuracy counter by the length of the label list to get
        # the accuracy in percent
        accuracy = accuracy / len(gold_labels)

        return accuracy, differences


def eval_subtitles(word_list, spacy_labels, stanza_labels):
    """Evaluate the subtitle-file: measure the concordance between the labels
    predicted by the SpaCy and Stanza language models, and return a list of
    all words which were annotated differently with the two models
    
    args: word_list (list of all words in the subtitle-file), spacy_labels 
    (list of all labels predicted by SpaCy), stanza_labels (list of all labels 
    predicted by Stanza)
    
    return: concordance (float, concordance of the lables predicted by the two
    language models), differences (list of lists, consisting of word and the 
    labels predicted by SpaCy and Stanza)"""

    if len(spacy_labels) == len(stanza_labels):
        concordance = 0.0
        diff_indexes = []
        differences = []

        spacy_labels = postprocess_labels(spacy_labels)

        for i in range(len(spacy_labels)):
            # if the current SpaCy and Stanza labels match add 1 to the 
            # concordance counter else add the current index to the
            # list of indexes of words with different SpaCy and Stanza
            # predictions
            if label_match(spacy_labels[i], stanza_labels[i]):
                concordance += 1.0
            else:
                diff_indexes.append(i)

        # create the list differences
        for index in diff_indexes:
            diff = [index, word_list[index], spacy_labels[index], stanza_labels[index]]
            differences.append(diff)

        # divide the concordance counter by the length of the label list to get
        # the concordance in percent
        concordance = concordance / len(spacy_labels)

        return concordance, differences
