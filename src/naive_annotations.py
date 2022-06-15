
# read processed data from spacy format file
from itertools import product


def read_spacy_file(file_name):
    all_data = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if lines[i].startswith('#id'):
            data = {'id': lines[i].replace('#id: ', '').strip(), 'text': lines[i+1].replace('#text: ', '').strip(),'persons':[],'locations':[]}
        elif lines[i].startswith('#'):
            continue
        elif lines[i] == '\n':
            all_data.append(data)
        else:
            line = lines[i].strip().split('\t')
            if line[7] == 'B':
                if line[8] == 'PERSON':
                    person = {'start': int(line[0]) -1,'end':int(line[0]) -1,  'type': 'PERSON'}
                    for j in range(i+1, len(lines)):
                        if lines[j].startswith('#') or lines[j] == '\n':
                            break
                        else:
                            line = lines[j].strip().split('\t')
                            if line[7] == 'I':
                                if line[8] == 'PERSON':
                                    person['end'] = int(line[0]) - 1
                            else:
                                break
                    person['text'] = ' '.join(data['text'].split()[person['start']:person['end']+1])
                    data['persons'].append(person)
                elif line[8] == 'GPE':
                    location = {'start': int(line[0]) -1,'end':int(line[0]) -1,  'type': 'GPE'}
                    for j in range(i+1, len(lines)):
                        if lines[j].startswith('#') or lines[j] == '\n':
                            break
                        else:
                            line = lines[j].strip().split('\t')
                            if line[7] == 'I':
                                if line[8] == 'GPE':
                                    location['end'] = int(line[0]) - 1
                            else:
                                break
                    location['text'] = ' '.join(data['text'].split()[location['start']:location['end']+1])
                    data['locations'].append(location)
                elif line[8] == 'LOC':
                    location = {'start': int(line[0]) -1,'end':int(line[0]) -1,  'type': 'LOC'}
                    for j in range(i+1, len(lines)):
                        if lines[j].startswith('#') or lines[j] == '\n':
                            break
                        else:
                            line = lines[j].strip().split('\t')
                            if line[7] == 'I':
                                if line[8] == 'LOC':
                                    location['end'] = int(line[0]) - 1
                            else:
                                break
                    location['text'] = ' '.join(data['text'].split()[location['start']:location['end']+1])
                    data['locations'].append(location)


    return all_data

def get_annotation_from_data(data):
    entities = []
    for d in data:
        if len(d['persons']) == 0 or len(d['locations']) == 0:
            continue
        else:
            for p in d['persons']:
                closest_loc = get_closest_location(d['locations'], p)
                entities.append({'id': d['id'], 'text': d['text'], 'person': p, 'location': closest_loc})

    return entities



def get_closest_location(locations, person):
    min_dist = 1000000
    closest_loc = None
    for l in locations:
        dist = abs(l['start'] - person['start'])
        if dist < min_dist:
            min_dist = dist
            closest_loc = l
    return closest_loc


def get_RE_from_file(file):
    RE = []
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        RE.append((line[0], line[1],  line[2]))
    return RE

def get_RE_from_entities(enteties):
    RE = []
    for e in enteties:
        RE.append((e['person']['text'], e['location']['text'],'Live_In'))
    return RE

def calc_precision_recall(RE, RE_pred):
    TP = 0
    FP = 0
    FN = 0
    for r in RE:
        if r in RE_pred:
            TP += 1
        else:
            FN += 1
    for r in RE_pred:
        if r not in RE:
            FP += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP) / (TP + FP + FN)
    return precision, recall, F1, accuracy

def tuples_annotations_from_data(data, RE='Live_In'):
    tuples = {}
    for d in data:
        if len(d['persons']) == 0 or len(d['locations']) == 0:
            continue
        tuples[d['id']] = []
        for p in d['persons']:
            for l in d['locations']:
                tuples[d['id']].append((p['text'],RE, l['text']))
    return tuples


data = read_spacy_file('data/Corpus.DEV.processed')

enteties = get_annotation_from_data(data[:])

RE = get_RE_from_entities(enteties)
t = tuples_annotations_from_data(data)

gold_RE = get_RE_from_file('annotations')


precision, recall, F1, accuracy = calc_precision_recall(RE, gold_RE)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1: ', F1)
print('Accuracy: ', accuracy)