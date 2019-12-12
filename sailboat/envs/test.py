import csv

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data.csv')

data = []

with open(filename, newline='') as csvfile:

    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    headerFound = False
    for row in reader:
        # print(': '.join(row))
        if (not headerFound):
            if (row[0] == 'ENDHEADER'):
                headerFound = True
        else :
            data.append([row[2], row[3]])
            print(row)
    
