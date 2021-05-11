import csv

disease_lsit = ['AF', 'BBB', 'TAC', 'normal']
file_path = 'D:/data/ECG/result/20210330/'
file_path_AF = file_path + 'AF.csv'
file_path_BBB = file_path + 'BBB.csv'
file_path_TAC = file_path + 'TAC.csv'
file_path_normal = file_path + 'normal.csv'


def read_file(file_path):
    result = []
    with open(file_path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            result.append([row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4])])
    return result


def add_data(data, some_result, groud_truth):
    assert groud_truth in disease_lsit
    for record in some_result:
        if record[0] not in data.keys():
            data[record[0]] = {'prob': [record[1], record[2], record[3], record[4]],
                               'ground_truth': [groud_truth]}
        else:
            assert data[record[0]]['prob'] == [record[1], record[2], record[3], record[4]]
            data[record[0]]['ground_truth'].append(groud_truth)
    return data


def guess(data):
    for item in data.values():
        item['guess'] = []
        if item['prob'][0] > 0.1:
            item['guess'].append('AF')
        if item['prob'][1] > 0.1:
            item['guess'].append('BBB')
        if item['prob'][2] > 0.1:
            item['guess'].append('TAC')
        if len(item['guess']) == 0:
            item['guess'].append('normal')
    return data


AF_result = read_file(file_path_AF)
BBB_result = read_file(file_path_BBB)
TAC_result = read_file(file_path_TAC)
normal_result = read_file(file_path_normal)

data = {}
data = add_data(data, AF_result, 'AF')
data = add_data(data, BBB_result, 'BBB')
data = add_data(data, TAC_result, 'TAC')
data = add_data(data, normal_result, 'normal')
data = guess(data)

tp_AF = 0
tp_BBB = 0
tp_TAC = 0
error = 0
# error_AF = 0
# error_BBB = 0
# error_TAC = 0
error_normal = 0
for key, value in data.items():
    if 'AF' in value['ground_truth'] and 'AF' in value['guess']:
        tp_AF += 1
    if 'BBB' in value['ground_truth'] and 'BBB' in value['guess']:
        tp_BBB += 1
    if 'TAC' in value['ground_truth'] and 'TAC' in value['guess']:
        tp_TAC += 1
    if 'normal' in value['ground_truth'] and 'normal' not in value['guess']:
        error_normal += 1
    if len(list(set(value['ground_truth']).intersection(set(value['guess'])))) == 0:
        error += 1

print(tp_AF/3000)
print(tp_BBB/3000)
print(tp_TAC/3000)
print((3000 - error_normal)/3000)
print(error/len(data))
