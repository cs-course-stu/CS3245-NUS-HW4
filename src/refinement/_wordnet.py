import nltk
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
terms = ['refin', 'can', 'token', 'queri', 'string', 'into',
         'term', 'and', 'comput', 'scienc', 'computing_machin', 'scientific_disciplin', 'tranquil']


def _generate_thesauri(word, terms):
    syn = set()
    syn.add(word.lower())
    for synset in wordnet.synsets(word.lower()):
        i = 0
        for lemma in synset.lemmas():
            tmp_stem = stemmer.stem(lemma.name().lower())
            # print(tmp_stem)
            if (tmp_stem in terms):
                syn.add(lemma.name())  # add the synonyms
                # print(lemma.name())
            if (len(syn) == 2):
                break
            i += 1
        if (len(syn) == 2):
            break
    return list(syn)


query = "  Computer Science"
tokens = [
    word for word in nltk.word_tokenize(query)
]
i = 0
query = ""
for token in tokens:
    tokens[i] = _generate_thesauri(token, terms)
    query = query + " ".join(tokens[i]) + " "
    i += 1
print(query)
# print(" ".join(tokens))
