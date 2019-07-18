import os
pat = "/storage/research/Intern19_v2/AutomatedDetectionWSI/LiverImages/"
#pat_1 = "/storage/research/Intern19_v2/AutomatedDetectionWSI/level_1/"
#pat_2 = "/storage/research/Intern19_v2/AutomatedDetectionWSI/level_2/"

a= os.walk(pat)
a = list(a)
l = []
for i in a[0][2]:
    if '.xml' in i or 'svs' in i or 'SVS' in i:
        continue
    else:
        l.append(i)
print(len(l))
#from pyslide import pyramid
from skimage import io
whole = {}
viable = {}
for i in l:
    p = os.path.join(pat,i)
    print(p)
    l_1 = io.imread(p)
    #print("l_1 loaded")
        
    
    d = i[:-4] # 01_01_0083_l_0
    print(d, l_1.shape)
    if 'whole' in d:
        whole[d] = l_1.shape
        print("whole")
    else:
        viable[d] = l_1.shape
        print("viable")
   
import pandas as pd
df = pd.DataFrame(whole)
#print(df.head)
df.to_csv("whole.csv")
df = pd.DataFrame(viable)
df.to_csv("viable.csv")