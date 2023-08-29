import spacy
import stanza
from time import perf_counter

# loading the spaCy and Stanza language models for English and Spanish
spacy_en = spacy.load("en_core_web_md")
spacy_es = spacy.load("es_core_news_md")
stanza_en = stanza.Pipeline("en", processors="tokenize,ner", package={"ner": ["conll03"]})
stanza_es = stanza.Pipeline("es", processors="tokenize,ner", package={"ner": ["conll02"]})

def load_europarl(filepath):
    """Load the data from a europarl conll02-file

    args: filepath (string, full path of the europarl file)

    return: words (list of all words in the file), labels (list of all labels), text (string of continuous text)

    note: the file path depends on the storage location of the file on the computer, and can vary from computer to computer
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

    return: text (string of continuous text without blank lines and line breaks)

    note: the file path depends on the storage location of the file on the computer, and can vary from computer to computer
    """

    text = ""

    with open(filepath, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                text += line.strip("\n") + " "

    return text

def ner(text, model, lang):
    """Process the given text, and return the list of recognized Named Entities

    args: text (string, continuous text), model (string, language model to be used i.e. spaCy or Stanza), language (string, language of the text)

    return: lists (list of all recognized Named Entities and non recognized Named Entities in the BIOES), the time to process the NER with the given NLP tool.

    note: when specifying the language, please use "en" for English and "es" for Spanish, and please write the names of the language models in lower case letters only
    """

    if model == "spacy":
        if lang == "en":
            doc = spacy_en(text)
        elif lang == "es":
            doc = spacy_es(text)
    elif model == "stanza":
        if lang == "en":
            #   Note regarding TIMER: The time will be different on each ussage (making each time output an estimation of the general performance time), 
            #       since the time is affected by the PC's performance and background application might "spike" the time/performance of the function.
            #       Instead we could use the modul "timeit", although we would need to perform stanza_en(text) twice or more,
            #       because timeit runs the method multiple times (without returning anything besides time),
            #       therefore it increasies the overall performance of the program.   
            doc = stanza_en(text)
            return [f'{token.text}\t{token.ner}\n' for sent in doc.sentences for token in sent.tokens], execution_time
        elif lang == "es":
            doc = stanza_es(text)
            return [f'{token.text}\t{token.ner}\n' for sent in doc.sentences for token in sent.tokens], execution_time
    return doc.ents

def compare_ner(first_ner, second_ner):
    '''Compares each line/word from both NER one at a time and compare them for differences. Placing the differences into a list.

    args: first_ner (the list of the first NER to be compared), second_ner (list of the second NER)

    return: differences (a list of all found differences in label(s) or word(s))   
    '''
    differences = []
    for i in range(len(first_ner)):
        if not first_ner[i] == second_ner[i]:
            differences.append(f"test: {first_ner[i]}stanza: {second_ner[i]}\nline: {i}\n")
    return differences

# --------------------------------------------------------------------------------------------------------------------------------------------------

""" ONLY FOR TEST PURPOSES, TO BE REMOVED LATER """

# TEST EUROPARL
#path_en = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Europarl Corpus/en-europarl.test.conll02"
path_en = "D:/OperaDownloads/Multilingual_NER-main/Multilingual_NER-main/Europarl Corpus/en-europarl.test.conll02"

w_en, l_en, t_en = load_europarl(path_en)
#print(w_en)
#print(l_en)
#print(t_en)

#path_es = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Europarl Corpus/es-europarl.test.conll02"
path_es = "D:/OperaDownloads/Multilingual_NER-main/Multilingual_NER-main/Europarl Corpus/es-europarl.test.conll02"

w_es, l_es, t_es = load_europarl(path_es)

#print(w_es)
#print(l_es)
#print(t_es)

#entities_sp = ner(t_es, "spacy", "es")
#print("spaCy: " + str(len(entities_sp)))
#for ent in entities_sp:
#    print(f"{ent.text:<25}{ent.label_:<15}")

entities_st, time = ner(t_es, "stanza", "es")

print(*entities_st, sep = '\n')
print(time)
#print("Stanza: " + str(len(entities_st)))
#for ent in entities_st:
#    print(f"{ent.text:<25}{ent.type:<15}")

# TEST SUBTITLES
#path_bttf = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Movie subtitles/Back To The Future (EN).txt"
path_bttf = "D:/OperaDownloads/Multilingual_NER-main/Multilingual_NER-main/Movie subtitles/El Hoyo (EN).txt"

#text = load_subtitles(path_bttf)
text2 = "Chris Manning teaches at Stanford University. He lives in the Bay Area."

#entities_sp = ner(text, "spacy", "en")
#for ent in entities_sp:
#    print(f"{ent.text:<25}{ent.label_:<15}")

entities_st = ner(text2, "stanza", "en")
#print(*entities_st, sep ='\n')
#for ent in entities_st:
#    print(f"{ent.text:<25}{ent.type:<15}")

#path_eh = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Movie subtitles/El Hoyo (ES).txt"
path_eh = "D:/OperaDownloads/Multilingual_NER-main/Multilingual_NER-main/Movie subtitles/El Hoyo (ES).txt"


#print(load_subtitles(path_eh))
