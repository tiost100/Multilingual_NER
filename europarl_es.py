from utils import load_europarl, ner, eval_europarl
from time import perf_counter

# Load the Spanish europarl-data
path_es = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Europarl Corpus/es-europarl.test.conll02"
words, labels, text = load_europarl(path_es)

"""---------------------------------SPACY-----------------------------------"""

# Perform the Named Entity Recognition with SpaCy and measure the time it takes
start_spacy = perf_counter()
entities_spacy = ner(text, "spacy", "es")
stop_spacy = perf_counter()
duration_spacy = stop_spacy - start_spacy

# Evaluate the labels predicted by SpaCy
accuracy_spacy, differences_spacy = eval_europarl(words, labels, entities_spacy, "spacy")

# Print the SpaCy results
with open("C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/europarl_es_spacy_eval.txt", "w") as outfile:
    outfile.write(f"Duration of the SpaCy NER in seconds: {round(duration_spacy, 3)} sec\n")
    outfile.write("\n")
    outfile.write(f"Accuracy of the SpaCy NER in percent: {round(accuracy_spacy * 100, 3)} %\n")
    outfile.write("\n")
    outfile.write("Differences:\n")
    outfile.write("Index   |Word                     |Gold Label     |Prediction     \n")
    outfile.write("------------------------------------------------------------------\n")
    for i in range(len(differences_spacy)):
        outfile.write(f"{differences_spacy[i][0]:<8}|{differences_spacy[i][1]:<25}|{differences_spacy[i][2]:<15}|{differences_spacy[i][3]:<15}\n")

print("SpaCy DONE!")

"""---------------------------------STANZA----------------------------------"""

# Perform the Named Entity Recognition with Stanza and measure the time it takes
start_stanza = perf_counter()
entities_stanza = ner(text, "stanza", "es")
stop_stanza = perf_counter()
duration_stanza = stop_stanza - start_stanza

# Evaluate the labels predicted by Stanza
accuracy_stanza, differences_stanza = eval_europarl(words, labels, entities_stanza, "stanza")

# Print the Stanza results
with open("C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/europarl_es_stanza_eval.txt", "w") as outfile:
    outfile.write(f"Duration of the Stanza NER in seconds: {round(duration_stanza, 3)} sec\n")
    outfile.write("\n")
    outfile.write(f"Accuracy of the Stanza NER in percent: {round(accuracy_stanza * 100, 3)} %\n")
    outfile.write("\n")
    outfile.write("Differences:\n")
    outfile.write("Index   |Word                     |Gold Label     |Prediction     \n")
    outfile.write("------------------------------------------------------------------\n")
    for i in range(len(differences_stanza)):
        outfile.write(f"{differences_stanza[i][0]:<8}|{differences_stanza[i][1]:<25}|{differences_stanza[i][2]:<15}|{differences_stanza[i][3]:<15}\n")

print("Stanza DONE!")