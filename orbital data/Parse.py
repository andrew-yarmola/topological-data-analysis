import pandas as pd

filename = input("Filename: ")
f = open(filename)

d = {'RA':[], 'dec':[], 'Earth Distance': [], 'Solar Distance':[], 'Elongation':[]}
for line in f:
    s = line.split()
    if s==[] or s[0] in ["Date", "(0", "h", "", "Geocentric", "00:00"]: continue
    RA = (float(s[2])+(float(s[3])+(float(s[4])/60))/60)*360/24
    dec= (float(s[5])+(float(s[6])+(float(s[7])/60))/60)
    EarthDistance = float(s[8])
    SolarDistance = float(s[9])
    sgn = 1
    if s[-1][-1] == 'W': sgn = -1
    Elongation = (float(s[-1][:-1]))*sgn
    
    d['RA'].append(RA)
    d['dec'].append(dec)
    d['Earth Distance'].append(EarthDistance)
    d['Solar Distance'].append(SolarDistance)
    d['Elongation'].append(Elongation)

df = pd.DataFrame.from_dict(d)

print(df)

df.to_csv(filename[:-4]+'.csv')
