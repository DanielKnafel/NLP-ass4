from collections import Counter

OUTSIDE = 'O'

# get corpus data and get all NER tags
def get_corpus_data(corpus_path):
    corpus_data = []
    with open(corpus_path, 'r') as corpus_file:
        l = []
        for line in corpus_file:
            if line == '\n':
                corpus_data.append(l)
                l = []
                continue
            if line.startswith('#'):
                continue
            l.append(line.strip().split('\t'))
        corpus_data.append(l)

    return corpus_data

def get_ner_tags(corpus_data):
    ner_tags = []
    for i,line in enumerate(corpus_data):
        for word in line:
            if word[7] != OUTSIDE:
                ner_tags.append((i,word[0], word[7],word[8]))

    return ner_tags

def get_types(ner_tags):
    types = set()
    for _,_,_,tag in ner_tags:
        types.add(tag)
    return types

def get_RE_from_file(file):
    with open(file, 'r') as f:
        file = f.readlines()
    objects = []
    subjects = []
    relations = []
    text = []
    idxs = []
    for line in file:
        l = line.strip().split('\t')
        idxs.append(l[0])
        objects.append(l[1])
        relations.append(l[2])
        subjects.append(l[3])
        text.append(l[4])
    return objects, subjects, relations,idxs

def write_RE_to_file(objects, subjects, relations, file, type = 'Live_In'):
    with open(file, 'w') as f:
        for i in range(len(objects)):
            if relations[i] == type:
                f.write(objects[i] + '\t' + subjects[i] + '\t' + relations[i] + '\n')



data = get_corpus_data('../data/Corpus.DEV.processed')
objects, subjects, relations ,idxs= get_RE_from_file('../data/DEV.annotations')
c = Counter(relations)
print(c)
ner_tags = get_ner_tags(data)
types = get_types(ner_tags)
types = {'GPE', 'PERSON', 'ORG','NERP', 'LOC','FAC'}
# counts = Counter(filter(lambda x: x[3], ner_tags))
counts = Counter(x[3] for x in filter(lambda x: x[3] in types, ner_tags))

print(counts)

write_RE_to_file(objects, subjects, relations, '../annotations', 'Live_In')