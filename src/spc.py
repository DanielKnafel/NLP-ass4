def file_to_dict(file):
    dic = {}
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        if line[0] not in dic:
            dic[line[0]] = [(line[1], line[2], line[3])]
        else:
            dic[line[0]].append((line[1], line[2], line[3]))
    return dic

def get_data_from_dict_and_file(dic, file):
    data = []
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        if line[0] in dic:

            for d in dic[line[0]]:
                data.append({'idx': line[0], 'sent1': line[1], 'sent2': line[2], 'label': line[3]})
    return data


train_annotations = file_to_dict('../data/TRAIN.annotations')
dev_annotations = file_to_dict('../data/DEV.annotations')