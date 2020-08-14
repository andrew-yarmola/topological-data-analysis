import cv2
import numpy as np
import plotly.express as px
import csv

#IMPORTANT! WE WILL REDUCE THIS
n = 30

#proteins that exist in our sample
proteins = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]
number_of_proteins = len(proteins)

#files dictionary
datafiles = {}
for i in proteins:
    datafiles[i] = []

with open("Data_in_use_" + str(n) + ".csv", 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if row[0] != "Id":
            datafiles[int(row[1])].append(row[0])    

#number of samples per protein
number_of_samples = len(datafiles[0])

######################################################################

# Read RGB image 
img = cv2.imread("data/" + datafiles[0][0] + "_green.png")  
  
# Output img with window name as 'img' 
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

######################################################################

shape = img.shape
labels = np.zeros(number_of_samples * number_of_proteins)
x_image, y_image = shape
images = np.zeros((number_of_samples * len(proteins), *shape))

######################################################################

# Load up images to the variable images as one long array
j = 0
for protein in datafiles:
    for i in range(len(datafiles[protein])):
        img = cv2.imread("data/" + datafiles[protein][i] + "_green.png")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        figure_size = 3 # the dimension of the x and y axis of the kernal.
        img = cv2.blur(img,(figure_size, figure_size))
        images[j] = img
        labels[j] = protein
        j = j+1
        
######################################################################

from gtda.images import HeightFiltration, RadialFiltration, \
DilationFiltration, ErosionFiltration, SignedDistanceFiltration, Binarizer

######################################################################

from gtda.pipeline import make_pipeline, Pipeline
from gtda.diagrams import BettiCurve
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceEntropy
from sklearn.pipeline import FeatureUnion

######################################################################

import itertools 

# Pipeline parameters
bin_thresholds = [((t+1) * .2) for t in range(4)]

directions = [np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 10)[:-1]]
x1 = int(x_image/4)
x2 = x1 * 2
x3 = x1 * 3
y1 = int(y_image/4)
y2 = y1 * 2
y3 = y1 * 3

centers = [np.array(p) for p in itertools.product((x1,x2,x3),(y1,y2,y3))]

######################################################################

# Vectorizations, we use L^2 for kernels by default and default n_bins
# Note, we are not acutally using kernel vectors, just their amplitudes for now
features = [('PE', PersistenceEntropy(n_jobs=-1))]

# Make filtrations
binned_steps = [('bin_{}'.format(t), Binarizer(threshold=t, n_jobs=-1)) for t in bin_thresholds]

######################################################################

filtrations = [('height_{:.2f}_{:.2f}'.format(*d), HeightFiltration(direction=d, n_jobs=-1)) for d in directions]
filtrations.extend([('radial_{}_{}'.format(*c), RadialFiltration(center=c, n_jobs=-1)) for c in centers])
filtrations.append(('dilation', DilationFiltration(n_jobs=-1)))
filtrations.append(('erosion', ErosionFiltration(n_jobs=-1)))
filtrations.append(('signed', SignedDistanceFiltration(n_jobs=-1)))

# Make pipelines
cubical_lower = [('cubical', CubicalPersistence(n_jobs=-1))]

partial_pipeline_steps = [cubical_lower]

for b, f in itertools.product(binned_steps, filtrations):
    partial_pipeline_steps.append([b, f, ('cubical', CubicalPersistence(n_jobs=-1))])
    
feature_pipelines = []
names = []
for s, f in itertools.product(partial_pipeline_steps, features):
    name = "{}_{}_{}".format(s[-3][0],s[-2][0], f[0]) if len(s) > 1 else "{}_{}".format(s[-1][0], f[0])
    names.append(name)
    feature_pipelines.append((name, Pipeline(s + [f])))
    
full_pipeline = Pipeline([('features', FeatureUnion(feature_pipelines))])

######################################################################

features = full_pipeline.fit_transform(images)

######################################################################

temp = np.zeros((*labels.shape, 1))
for label in range(len(labels)):
    temp[label][0] = labels[label]

concat_labels = temp
# labels vector ready for concatenation

######################################################################

import pandas as pd
import math
number_of_features = len(features[0])

# set up numpy array to convert
to_convert = np.append(features, concat_labels, axis=1)

######################################################################
df = pd.DataFrame(data=to_convert)
df.to_csv('MANYfiltrations.csv', encoding='utf-8')
######################################################################
