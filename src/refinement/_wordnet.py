import nltk
from nltk.corpus import wordnet


def _generate_thesauri(word):
    syn = set()
    for synset in wordnet.synsets(word.lower()):
        for lemma in synset.lemmas():
            syn.add(lemma.name())  # add the synonyms
        if (len(syn) == 3):
            break    
    return syn


rst = _generate_thesauri("Worse")
print(rst)
