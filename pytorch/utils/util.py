def process_label_file(label_file):
    dictionary, reverse_dict = {}, {}
    with open(label_file, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            dictionary[line[0]] = int(line[1])
            reverse_dict[int(line[1])] = line[0]
    return dictionary, reverse_dict