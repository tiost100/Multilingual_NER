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

# --------------------------------------------------------------------------------------------------------------------------------------------------

""" ONLY FOR TEST PURPOSES, TO BE REMOVED LATER """

# TEST EUROPARL
path_en = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Europarl Corpus/en-europarl.test.conll02"

w_en, l_en, t_en = load_europarl(path_en)
#print(w_en)
#print(l_en)
#print(t_en)

path_es = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Europarl Corpus/es-europarl.test.conll02"

w_es, l_es, t_es = load_europarl(path_es)
#print(w_es)
#print(l_es)
#print(t_es)

# TEST SUBTITLES
path_bttf = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Movie subtitles/Back To The Future (EN).txt"

print(load_subtitles(path_bttf))

path_eh = "C:/Users/Tim.O/Documents/Studium/4. Semester/Advanced Python for NLP/ABSCHLUSSPROJEKT/Movie subtitles/El Hoyo (ES).txt"

#print(load_subtitles(path_eh))