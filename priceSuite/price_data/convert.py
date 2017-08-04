f = open('Louisiana.dat')

l = f.readlines()

data = []
for line in l:
    data.append(float(line.split(',')[0]))

w = open('Louisiana.csv', 'w')
#w.write('\"index\", \"price\"\n')
w.write('\"price\"\n')
i=1
for item in data:
    #w.write(str(i)+','+str(item)+'\n')
    w.write(str(item)+'\n')
    i=i+1
w.close()
    
