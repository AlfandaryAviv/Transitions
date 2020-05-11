import os
import csv



p=os.getcwd()+"/logs" #
all_dicts = []
for subdir, dirs, files in os.walk(p):
    for file in files:
        if "log" in file:
            f=open(subdir+"/"+file,'r')
            lines=f.readlines()
            b=5
            if "0393" in lines[0]:
                print(lines[0])
                
                
