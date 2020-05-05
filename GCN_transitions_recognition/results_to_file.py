import os
import csv

'''
p=os.getcwd()+"/logs_last"
all_dicts = []
for subdir, dirs, files in os.walk(p):
    for file in files:
        if "log" in file:
            f=open(subdir+"/"+file,'r')
            lines=f.readlines()
            params = lines[0].split(" ")
            results = lines[21].split(" ")
            dict = {"lr": params[8], "dropout":params[11], "regularization": params[14], "temp_pen":params[17], "optimizer":params[20],
                          "loss_test": results[7], "acc_test":results[9].split('\n')[0]}
            all_dicts.append(dict)
            f.close()



csv_columns = ['lr','dropout','regularization','temp_pen' , 'optimizer', 'loss_test', 'acc_test']

csv_file = "results.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for dict in all_dicts:
            writer.writerow(dict)
except IOError:
    print("I/O error")
    
    '''

p=os.getcwd()+"/logs"
all_dicts = []
for subdir, dirs, files in os.walk(p):
    for file in files:
        if "log" in file:
            f=open(subdir+"/"+file,'r')
            lines=f.readlines()
            b=5
            if "0393" in lines[0]:
                print(lines[0])
