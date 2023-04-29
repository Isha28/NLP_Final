import csv
	
with open('train.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    with open('train.tsv', 'w', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        for row in csv_reader:
            tsv_writer.writerow(row)

with open('test.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    with open('test.tsv', 'w', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        for row in csv_reader:
            tsv_writer.writerow(row)
with open('dev.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    with open('dev.tsv', 'w', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        for row in csv_reader:
            tsv_writer.writerow(row)

