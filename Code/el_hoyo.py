from utils import load_subtitles, ner, eval_subtitles
from time import perf_counter
import os

# Get the full path of the directory where the current file is located
dir_path = os.path.dirname(os.path.abspath(__file__))

# get the parent directory path
parent_dir_path = os.path.dirname(dir_path)
parent_dir_path = parent_dir_path.replace('\\','/')

path_en = f"{parent_dir_path}/Data/Movie subtitles/El Hoyo (EN).txt"
path_es = f"{parent_dir_path}/Data/Movie subtitles/El Hoyo (ES).txt"

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
with open(f"{parent_dir_path}/Data/Evaluation Results/el_hoyo_en_eval.txt", "w", encoding="latin-1") as outfile:
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
with open(f"{parent_dir_path}/Data/Evaluation Results/el_hoyo_es_eval.txt", "w", encoding="latin-1") as outfile:
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