import argparse
import json
import random
import re


def read_file(input_file):
    data = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    lives_in = 0
    for id, line in enumerate(lines):
        line = line.strip().split('\t')
        sent1 = line[4][1:-1]
        sent2 = line[1] + " lives in " + line[3]
        label = int(line[2] == 'Live_In')
        lives_in += label
        data.append({'idx': id, 'sent1': sent1, 'sent2': sent2, 'label': label})
    print("Lives in:", lives_in)
    return data


def read_spacy_file(file_name):
    persons = []
    locations = []
    all_data = {}
    with open(file_name, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if lines[i].startswith('#id'):
            data = {'id': lines[i].replace('#id: ', '').strip(), 'text': [], 'persons': [], 'locations': []}
        elif lines[i].startswith('#'):
            for j in range(i + 1, len(lines)):
                if lines[j] == '\n':
                    break
                line = lines[j].strip().split('\t')
                data['text'].append(line[1])
        elif lines[i] == '\n':
            data['text'] = ' '.join(data['text'])
            all_data[data['id']] = data
        else:
            line = lines[i].strip().split('\t')

            if line[7] == 'B':
                if line[8] == 'PERSON':
                    person = {'start': int(line[0]) - 1, 'end': int(line[0]) - 1, 'type': 'PERSON'}
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith('#') or lines[j] == '\n':
                            break
                        else:
                            line = lines[j].strip().split('\t')
                            if line[7] == 'I':
                                if line[8] == 'PERSON':
                                    person['end'] = int(line[0]) - 1
                            else:
                                break
                    person['text'] = ' '.join(data['text'][person['start']:person['end'] + 1]).replace(' - ', '-')
                    data['persons'].append(person)
                    persons.append(person['text'])
                elif line[8] == 'GPE' or line[8] == 'LOC' or line[8] == 'NORP':
                    type = line[8]
                    location = {'start': int(line[0]) - 1, 'end': int(line[0]) - 1, 'type': type}
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith('#') or lines[j] == '\n':
                            break
                        else:
                            line = lines[j].strip().split('\t')
                            if line[7] == 'I':
                                if line[8] == type:
                                    location['end'] = int(line[0]) - 1
                            else:
                                break
                    location['text'] = ' '.join(data['text'][location['start']:location['end'] + 1]).replace(' - ', '-')
                    data['locations'].append(location)
                    locations.append(location['text'])

    return all_data, persons, locations


def tuples_annotations_from_data(data, RE='lives in'):
    tuples = {}
    for dk, dv in data.items():
        if len(dv['persons']) == 0 or len(dv['locations']) == 0:
            continue
        tuples[dk] = []
        for p in dv['persons']:
            for l in dv['locations']:
                tuples[dk].append((p['text'], RE, l['text'], (p['start'], p['end']), (l['start'], l['end'])))
    return tuples


def json_to_file(data, output_file):
    with open(output_file, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')


def get_label(t, sentences, RE='Live_In'):
    for s in sentences:
        if s[1] == t[0] and s[2] == RE and s[3] == t[2]:
            return 1
    return 0


RE_sentences = []


def get_dataset_from_file_and_tuples(file, processed_data, tuples, RE='Live_In', marker=False):
    global RE_sentences
    data = []
    with open(file, 'r') as f:
        lines = f.readlines()
    idx = 0
    last_line = ''
    RE_sentences = [line.split('\t') for line in lines if line.split('\t')[2] == RE]

    for i, line in enumerate(lines):
        line = line.strip().split('\t')
        if line[0] == last_line:
            continue
        if line[0] in tuples.keys():
            sent_lines = [lines[j] for j in range(i, len(lines)) if lines[j].startswith(line[0] + '\t')]
            sent_lines = [l.strip().split('\t') for l in sent_lines]
            for t in tuples[line[0]]:
                sent1 = processed_data[line[0]]['text']
                # sent1 = line[4][2:-1]
                if marker:
                    sent1 = sent1.split()
                    if t[3][0] < t[4][0]:
                        sent1.insert(t[3][0], '<p>')
                        sent1.insert(t[3][1] + 2, '</p>')
                        sent1.insert(t[4][0] + 2, '<l>')
                        sent1.insert(t[4][1] + 4, '</l>')
                    else:
                        sent1.insert(t[4][0], '<l>')
                        sent1.insert(t[4][1] + 2, '</l>')
                        sent1.insert(t[3][0] + 2, '<p>')
                        sent1.insert(t[3][1] + 4, '</p>')
                    sent1 = ' '.join(sent1)

                data.append({'id': line[0], 'idx': idx, 'sent1': sent1.replace(' - ', '-'), 'sent2': " ".join(t[0:3]),
                             'label': get_label(t, sent_lines)})
                idx += 1
        last_line = line[0]
    return data


def get_True_False_RE_from_files(text_file, annotations_file):
    data = []
    with open(annotations_file, 'r') as f:
        lines = f.readlines()
    sentences = set([line.strip().split('\t')[0] for line in lines])
    with open(text_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        idx, sent = line.strip().split(maxsplit=1)
        data.append({'idx': i, 'sent': sent, 'label': int(idx in sentences)})
    return data


def get_data_biases(data):
    trues, falses = 0, 0
    for d in data:
        if d['label'] == 1:
            trues += 1
        else:
            falses += 1
    return trues / len(data), falses / len(data)


def generate_sample(data, person, location, person_loc, location_loc):
    text = data['text'].split()
    if person_loc[0] > location_loc[0]:
        text = text[:person_loc[0]] + text[person_loc[1] + 1:]
        text.insert(person_loc[0], person)
        text = text[:location_loc[0]] + text[location_loc[1] + 1:]
        text.insert(location_loc[0], location)
    else:
        text = text[:location_loc[0]] + text[location_loc[1] + 1:]
        text.insert(location_loc[0], location)
        text = text[:person_loc[0]] + text[person_loc[1] + 1:]
        text.insert(person_loc[0], person)
    return ' '.join(text)


def fix_biases(data, data1, biases, persons, locations, marked=False):
    need_to_generate = int(len(data) * biases[1] - len(data) * biases[0])
    trues_data = [d for d in data if d['label'] == 1]
    idx = data[-1]['idx'] + 1
    while need_to_generate > 0:
        for d in trues_data:
            if need_to_generate == 0:
                break
            for d1 in data1.values():
                text = d['sent1'] if not marked else d['sent1'].replace('<p> ', '').replace('</p> ', '').replace('<l> ',
                                                                                                                 '').replace(
                    '</l> ', '')
                if text == d1['text']:
                    if random.random() < 0.5:
                        person_loc = [(x['start'], x['end']) for x in d1['persons'] if
                                      x['text'] == re.split(' lives in ', d['sent2'])[0]][0]
                        location_loc = [(x['start'], x['end']) for x in d1['locations'] if
                                        x['text'] == re.split(' lives in ', d['sent2'])[1]][0]
                        person = '<p> ' + random.choice(persons) + '</p> ' if marked else random.choice(persons)
                        location = '<l> ' + random.choice(locations) + '</l> ' if marked else random.choice(locations)
                        sent = generate_sample(d1, person, location, person_loc, location_loc)
                        data.append(
                            {'idx': idx, 'sent1': sent, 'sent2': " ".join([person, 'lives in', location]), 'label': 1})
                        idx += 1
                        need_to_generate -= 1
                        if need_to_generate == 0:
                            break
    random.shuffle(data)
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_processed', default='train_processed', help='train processed data')
    parser.add_argument('--dev_processed', default='dev_processed', help='dev processed data')
    parser.add_argument('--train_annotations', default='train_annotations', help='train annotated data')
    parser.add_argument('--dev_annotations', default='dev_annotations', help='dev annotated')
    parser.add_argument('--output_files_dir', default='')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    train_data1, persons, locations = read_spacy_file(args.train_processed)
    train_tuples = tuples_annotations_from_data(train_data1)

    dev_data1, persons_dev, locations_dev = read_spacy_file(args.dev_processed)
    dev_tuples = tuples_annotations_from_data(dev_data1)

    train_data = get_dataset_from_file_and_tuples(args.train_annotations, train_data1, train_tuples)
    dev_data = get_dataset_from_file_and_tuples(args.dev_annotations, dev_data1, dev_tuples)

    marked_train_data = get_dataset_from_file_and_tuples(args.train_annotations, train_data1, train_tuples, marker=True)
    marked_dev_data = get_dataset_from_file_and_tuples(args.dev_annotations, dev_data1, dev_tuples, marker=True)

    train_data_from_annotations = read_file(args.dev_annotations)
    dev_data_from_annotations = read_file(args.dev_annotations)

    # biases = get_data_biases(train_data)
    # train_data = fix_biases(train_data, train_data1, biases, persons, locations)
    # marked_train_data = fix_biases(marked_train_data, train_data1, biases, persons, locations, marked=True)

    biases = get_data_biases(dev_data)
    json_to_file(dev_data, args.output_files_dir + '/DEV.json')
    dev_data = fix_biases(dev_data, dev_data1, biases, persons_dev, locations_dev)
    json_to_file(marked_dev_data, args.output_files_dir + '/DEV.marked.json')
    marked_dev_data = fix_biases(marked_dev_data, dev_data1, biases, persons_dev, locations_dev, marked=True)



    json_to_file(train_data, args.output_files_dir + '/TRAIN.json')
    json_to_file(dev_data, args.output_files_dir + '/DEV.fix_biased.json')

    json_to_file(marked_train_data, args.output_files_dir + '/TRAIN.marked.json')
    json_to_file(marked_dev_data, args.output_files_dir + '/DEV.marked_fix_biased.json')

    json_to_file(train_data_from_annotations, args.output_files_dir + '/TRAIN.from_annotations.json')
    json_to_file(dev_data_from_annotations, args.output_files_dir + '/DEV.from_annotations.json')
