# Authors: Everett Wilson and Andrew Walker
# Assignment: NLP Undergrad Project - Part 5, HMM POS tagging

from nltk.corpus import treebank
from nltk.tag import hmm
import numpy as np

# A modified Hidden Markov Model POS tagger, that allows for modified POS tags
class HmmTagger2:
    # Constructor
    # Parameters:
    # labeled_sequences - an array of tuples representing a corpus of tagged sentences to train the model with
    # substitute - a dictionary of POS substitutions to make
    def __init__(self, labeled_sequences, substitute):
        self.substitute = substitute
        trainer = hmm.HiddenMarkovModelTrainer()
        self.my_tagger = trainer.train_supervised(self.process_lines(labeled_sequences))

    # Preprocess an array of tagged sentence tuples to use our substitution dictionary
    # Unfortunately, have to build new arrays, since in-place tuple editing is not allowed
    # Parameters:
    # lines - array of tuples from a tagged corpus of the format (word, POS)
    def process_lines(self, lines):
        new_lines = []
        for line in lines:
            new_line = []
            for tup in line:
                elem1 = tup[1]
                if tup[1] in self.substitute:
                    elem1 = self.substitute[tup[1]]
                new_line.append( (tup[0], elem1) )
            new_lines.append(new_line)
        return new_lines

    # Return trained tagger
    def tagger(self):
        return self.my_tagger

# Set up substitution dictionary
substitutions = {
    "VB": "VBE",
    "VBD": "VBE",
    "VBG": "VBE",
    "VBN": "VBE",
    "VBP": "VBE",
    "VBZ": "VBE",
    "VH": "VH",
    "VHD": "VH",
    "VHG": "VH",
    "VHN": "VH",
    "VHP": "VH",
    "VHZ": "VH",
    "VV": "V",
    "VVD": "V",
    "VVG": "V",
    "VVN": "V",
    "VVP": "V",
    "VVZ": "V",
    "JJ": "J",
    "JJR": "J",
    "JJS": "J",
    "NN": "N",
    "NNS": "N",
    "NP": "N",
    "NPS": "N",
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV"
}
# Instantiate, train, and get tagger
tagged_sents = treebank.tagged_sents()
training_size = 3000
hmm2 = HmmTagger2(treebank.tagged_sents()[:training_size], substitutions)
tagger = hmm2.tagger()
# Compare our output with given results, tally up accuracy
correct = 0
total = 0
for idx, line in enumerate(treebank.sents()[training_size:]):
    corpus_sent = tagged_sents[training_size+idx]
    our_sent = tagger.tag(line)
    print("Corpus sentence:", corpus_sent)
    print("Our tagged sentence:", our_sent)
    if len(corpus_sent) != len(our_sent):
        print("Corpus sentence and our sentence did not have matching length, skipping accuracy measurement...")
    else:
        for index, tup in enumerate(our_sent):
            total = total + 1
            tag = corpus_sent[index][1]
            if tag in substitutions.keys():
                tag = substitutions[tag]
            if tup[1] == tag:
                correct = correct + 1
print("Total accuracy: ", (correct / total) * 100, "%")