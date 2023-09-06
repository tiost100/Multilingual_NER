# Multilingual Named Entity Recognition with SpaCy and Stanza

## Table of Contents
1. [General Info](#general-info)
2. [Status](#status)
3. [Requirements](#requirements)
4. [Download](#download)
5. [Usage](#usage)
6. [Technologies/Sources](#technologiessources)
7. [Licence](#licence)

## General Info:
In this project, we investigate the performance of the NLP tools SpaCy and Stanza on Named Entity Recognition (NER) in English and Spanish. In the first step, we compare the accuracy and runtime of the NER methods of the two tools on parallel annotated transcriptions of European Parliament sessions (Europarl corpus). These were manually annotated word by word in CoNLL 2002/2003 format; there are 4 entity types: PER, LOC, ORG and MISC. Then, on selected unannotated movie subtitle files in English and Spanish (Open Subtitles), we analyze the level of concordance in SpaCy and Stanza predictions and what was annotated differently in the two NLP tools.

The utils.py file contains all the auxiliary methods needed to load the data, perform Named Entity Recognition, and evaluate the results. In the remaining Python files the evaluations of the different files are executed; the results are loaded into the .txt files with the respective file names, such as europarl_en_eval.txt. You can use all Python files except utils.py as a starting point, because the different programs do not build on each other.

For further information, please see the report.

## Status:
- Europarl data: DONE
- Subtitles: IN PROGRESS

## Requirements:
In order for the programs to work on your computer, please download the following:
- Python version 3.11.4
- SpaCy:<pre>pip install -U pip setuptools wheel <br>pip install -U spacy </pre>
  - SpaCy language models:
    <pre>python -m spacy download en_core_web_md    # for English <br>python -m spacy download es_core_news_md   # for Spanish</pre>
- Stanza:<pre>pip install stanza</pre>

## Download:
To be able to run the project on your own computer, please clone this GitHub repository by running the following command in your terminal; you have to run the terminal as administrator:
<pre>git clone https://github.com/tiost100/Multilingual_NER</pre>

## Usage:

## Technologies/Sources:
We used following tools resp. corpora in this project:
- SpaCy NLP tool:
  - https://spacy.io/
  - Matthew Honnibal, Ines Montani, Sofie Van Landeghem, Adriane Boyd. 2020. spaCy: Industrial-strength Natural Language Processing in Python. 2020
- Stanza NLP tool:
  - https://stanfordnlp.github.io/stanza/
  - Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In Association for Computational Linguistics (ACL) System Demonstrations. 2020.
- Evaluation Corpus for Named Entity Recognition using Europarl:
  - https://github.com/ixa-ehu/ner-evaluation-corpus-europarl
  - Rodrigo Agerri, Yiling Chung, Itziar Aldabe, Nora Aranberri, Gorka Labaka and German Rigau (2018). Building Named Entity Recognition Taggers via Parallel Corpora. In Proceedings of the 11th Language Resources and Evaluation Conference (LREC 2018), 7-12 May, 2018, Miyazaki, Japan.
  - Europarl: A Parallel Corpus for Statistical Machine Translation, Philipp Koehn, MT Summit 2005.
- OpenSubtitles: <br />https://www.opensubtitles.org/
- Happy Scribe (a tool that converts subtitles in .srt format to .txt file): <br />https://www.happyscribe.com/de/untertitel-tools/untertitel-converter

## Licence:
We are not aware of any copyright restrictions of the material.
