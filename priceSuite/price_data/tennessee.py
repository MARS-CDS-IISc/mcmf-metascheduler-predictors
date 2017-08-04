f = open('Tennessee.csv','w')

f.write("\"price\"\n")

onpeak = 124.35
offpeak = 112.72
industrial = 74.0
for i in range(720):
    f.write(str(industrial) + '\n')
    '''
    if (i+1) % 24 >= 12 and (i+1) % 25 <= 12+8:
        f.write(str(onpeak) + '\n')
    else:
        f.write(str(offpeak) + '\n')
    '''
f.close()
