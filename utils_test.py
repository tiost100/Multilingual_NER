import spacy
import stanza
from spacy.tokens import Doc

# SOURCE: https://stackoverflow.com/questions/65160277/spacy-tokenizer-with-only-whitespace-rule
# by user "Sofie VL"
# START 
class WhitespaceTokenizer(object):
    """Tokenizer splitting text on whitespaces only, for processing of texts
    in English with the SpaCy language model. With the default SpaCy tokenizer,
    the length of the doc was bigger than the length of the list with the gold
    labels.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

# loading the spaCy and Stanza language models for English and Spanish
spacy_en = spacy.load("en_core_web_md")
spacy_en.tokenizer = WhitespaceTokenizer(spacy_en.vocab)
# END

spacy_es = spacy.load("es_core_news_md")
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
                label = label[:-1]

                words.append(parts[0])
                labels.append(label)

    text = " ".join(words)

    return words, labels, text


def load_subtitles(filepath):
    """Load movie subtitle txt-file, and remove blank lines and line breaks

    args: filepath (string, full path of the subtitle file)

    return: text (string of continuous text without blank lines and line 
    breaks)

    note: the file path depends on the storage location of the file on the 
    computer, and can vary from computer to computer
    """

    text = ""

    with open(filepath, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                text += line.strip("\n") + " "

    return text

def ner(text, model, lang):
    """Process the given text, and return the list of predicted labels

    args: text (string, continuous text), model (string, language model to be 
    used i.e. spaCy or Stanza), language (string, language of the text)

    return: list of all predicted labels (including recognized Named Entities
    as well as words which are not Named Entities) in the BIOES format

    note: when specifying the language, please use "en" for English and "es" 
    for Spanish, and please write the names of the language models in lower 
    case letters only
    """

    if model == "spacy":
        if lang == "en":
            doc = spacy_en(text)
            preds = [doc[i].ent_iob_ + "-" + doc[i].ent_type_ for i in range(len(doc))]
        elif lang == "es":
            doc = spacy_es(text)
            preds = [doc[i].ent_iob_ + "-" + doc[i].ent_type_ for i in range(len(doc))]
    elif model == "stanza":
        if lang == "en":
            doc = stanza_en(text)
            preds = [token.ner for sent in doc.sentences for token in sent.tokens]
        elif lang == "es":
            doc = stanza_es(text)
            preds = [token.ner for sent in doc.sentences for token in sent.tokens]
    return preds
        

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

    accuracy = 0.0
    differences = []
    
    if model == "spacy":
        # preprocessing
        pass
    elif model == "stanza":
        pass

    return accuracy, differences


def eval_subtitles(spacy_labels, stanza_labels):
    """Evaluate the subtitle-file: measure the concordance between the labels
    predicted by the SpaCy and Stanza language models, and return a list of
    all words which were annotated differently with the two models
    
    args: spacy_labels (list of all labels predicted by SpaCy), stanza_labels
    (list of all labels predicted by Stanza)
    
    return: concordance (float, concordance of the lables predicted by the two
    language models), differences (list of lists, consisting of word and the 
    labels predicted by SpaCy and Stanza)"""

    concordance = 0.0
    differences = []

    return concordance, differences


# TEST
#path_en = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Europarl Corpus/en-europarl.test.conll02"
#path_es = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Europarl Corpus/es-europarl.test.conll02"
#w_en, l_en, t_en = load_europarl(path_en)
#w_es, l_es, t_es = load_europarl(path_es)

#entities_sp = ner(t_en, "spacy", "en")
#print(len(w_en))
#print(len(entities_sp))
#for i in range(len(w_en)):
    #print(w_en[i] + "\t" + l_en[i] + "\t" + entities_sp[i])
    #if w_en[i] != entities_sp[i]:
    #    print("falsch: " + w_en[i] + " " + entities_sp[i])
#print("FERTIG")

#entities_sp = ner(t_es, "spacy", "es")
#print(len(w_es))
#print(len(entities_sp))
#for i in range(len(w_es)):
    #print(w_es[i] + "\t" + l_es[i] + "\t" + entities_sp[i])
    #if w_es[i] != entities_sp[i]:
    #    print("falsch: " + w_es[i] + " " + entities_sp[i])
#print("FERTIG")

#entities_st = ner(t_en, "stanza", "en")
#print(len(w_en))
#print(len(entities_st))
#print("Word\tGoldLabel\tPrediction")
#for i in range(len(w_en)):
    #print(w_en[i] + "\t" + l_en[i] + "\t" + entities_st[i])
#print("richtig")
#print(*entities_st, sep = '\n')

#entities_st = ner(t_es, "stanza", "es")
#print(len(w_es))
#print(len(entities_st))
#for i in range(len(w_es)):
    #print(w_es[i] + "\t" + l_es[i] + "\t" + entities_st[i])
#print(*entities_st, sep = '\n')

""" 
PROBLEM: Invalid empty string in whitespace tokenizer

path_eh_en = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Movie subtitles/El Hoyo (EN).txt"
path_eh_es = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Movie subtitles/El Hoyo (ES).txt"

text_en = load_subtitles(path_eh_en)
text_es = load_subtitles(path_eh_es)

entities_sp_en = ner(text_en, "spacy", "en")
entities_st_en = ner(text_en, "stanza", "en")

print(len(entities_sp_en))
print(len(entities_st_en))

entities_sp_es = ner(text_en, "spacy", "es")
entities_st_es = ner(text_en, "stanza", "es")

print(len(entities_sp_es))
print(len(entities_st_es)) """