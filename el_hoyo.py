from utils import load_subtitles, ner, eval_subtitles
from time import perf_counter

path_en = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Movie subtitles/El Hoyo (EN).txt"
path_es = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Movie subtitles/El Hoyo (ES).txt"

words_en, text_en = load_subtitles(path_en)
words_es, text_es = load_subtitles(path_es)

"""--------------------------------ENGLISH----------------------------------"""

# Perform the Named Entity Recognition with SpaCy and measure the time it takes
start_spacy_en = perf_counter()
entities_spacy_en = ner(text_en, "spacy", "en")
stop_spacy_en = perf_counter()
duration_spacy_en = stop_spacy_en - start_spacy_en

# Perform the Named Entity Recognition with Stanza and measure the time it takes
start_stanza_en = perf_counter()
entities_stanza_en = ner(text_en, "stanza", "en")
stop_stanza_en = perf_counter()
duration_stanza_en = stop_stanza_en - start_stanza_en

# Evaluate the labels predicted by SpaCy
concordance_en, differences_en = eval_subtitles(words_en, entities_spacy_en, 
                                                entities_stanza_en)

# Print the SpaCy results
with open("C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/el_hoyo_en_eval.txt", "w", encoding="latin-1") as outfile:
    outfile.write(f"Duration of the SpaCy NER in seconds:  {round(duration_spacy_en, 3)} sec\n")
    outfile.write(f"Duration of the Stanza NER in seconds: {round(duration_stanza_en, 3)} sec\n")
    outfile.write("\n")
    outfile.write(f"Concordance of the SpaCy and Stanza in percent: {round(concordance_en * 100, 3)} %\n")
    outfile.write("\n")
    outfile.write("Differences:\n")
    outfile.write("Index   |Word                     |Spacy Label    |Stanza Label   \n")
    outfile.write("------------------------------------------------------------------\n")
    for i in range(len(differences_en)):
        outfile.write(f"{differences_en[i][0]:<8}|{differences_en[i][1]:<25}|{differences_en[i][2]:<15}|{differences_en[i][3]:<15}\n")

print("English DONE!")

"""--------------------------------SPANISH----------------------------------"""

# Perform the Named Entity Recognition with SpaCy and measure the time it takes
start_spacy_es = perf_counter()
entities_spacy_es = ner(text_es, "spacy", "es")
stop_spacy_es = perf_counter()
duration_spacy_es = stop_spacy_es - start_spacy_es

# Perform the Named Entity Recognition with Stanza and measure the time it takes
start_stanza_es = perf_counter()
entities_stanza_es = ner(text_es, "stanza", "es")
stop_stanza_es = perf_counter()
duration_stanza_es = stop_stanza_es - start_stanza_es

# Evaluate the labels predicted by SpaCy
concordance_es, differences_es = eval_subtitles(words_es, entities_spacy_es, 
                                                entities_stanza_es)

# Print the SpaCy results
with open("C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/el_hoyo_es_eval.txt", "w", encoding="latin-1") as outfile:
    outfile.write(f"Duration of the SpaCy NER in seconds:  {round(duration_spacy_es, 3)} sec\n")
    outfile.write(f"Duration of the Stanza NER in seconds: {round(duration_stanza_es, 3)} sec\n")
    outfile.write("\n")
    outfile.write(f"Concordance of SpaCy and Stanza in percent: {round(concordance_es * 100, 3)} %\n")
    outfile.write("\n")
    outfile.write("Differences:\n")
    outfile.write("Index   |Word                     |Spacy Label    |Stanza Label   \n")
    outfile.write("------------------------------------------------------------------\n")
    for i in range(len(differences_es)):
        outfile.write(f"{differences_es[i][0]:<8}|{differences_es[i][1]:<25}|{differences_es[i][2]:<15}|{differences_es[i][3]:<15}\n")

print("Spanish DONE!")