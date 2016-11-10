
# coding: utf-8

# In[20]:
import sys
from ijson import parse
import ijson
import csv


# In[31]:

#fw = open('test.txt','a')
fieldnames = ['entityId', 'charset', 'entity', 'alias']
languages = ['bn','as', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'sa', 'ta', 'te','en']
statistics = {}

csvfile = open(sys.argv[2], 'a') 
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#writer.writeheader()


f = open(sys.argv[1])
for item in ijson.items(f, "item"):
   
    id_ = item['id']
    labels=item['labels']
    if('en' in item['labels'].keys()):
    	entity_ = item['labels']['en']['value'].encode('utf-8')
    	#print labels
    	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    	
    	for label in labels:
        	charset_ = labels[label]['language'].encode('utf-8')
        
        	if charset_ in languages:
            		if charset_ not in statistics.keys():
                		statistics[charset_]=1
            		else:
                		statistics[charset_]+=1
            		alias_ = labels[label]['value'].encode('utf-8')
            
            		#writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            		writer.writerow({fieldnames[0]:id_, fieldnames[1]:charset_, fieldnames[2]:entity_, fieldnames[3]:alias_ })        
    #break

csvfile.close()
fw = open(sys.argv[3],'a')
for key in statistics.keys():
    fw.write(key+" "+str(statistics[key])+"\n")
fw.close()
#print statistics        
#fw.close()


# In[ ]:



