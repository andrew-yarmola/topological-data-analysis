import cv2
import numpy as np
import plotly.express as px
import itertools
from gtda.images import Binarizer, HeightFiltration, Inverter, DilationFiltration, ErosionFiltration
from gtda.pipeline import Pipeline
from gtda.pipeline import make_pipeline
from gtda.diagrams import BettiCurve
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import Amplitude

def generate_features(img_file):
    """
    Applies image to a custom feature_pipline
    """

    img = cv2.imread(img_file)  
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blur the image to reduce noise
    figure_size = 9 # the dimension of the x and y axis of the kernal.
    img = cv2.blur(img,(figure_size, figure_size))
    shape = img.shape
    images = np.zeros((1, *shape))
    images[0] = img
    
    features = []
    for p in pipeline1(images):
        for f in p.fit_transform(images)[0]:
            features.append(f)
    return features


def pipeline1(images):
    """
    Binarizer --> Height Filtration, Erosion Filtration, Dilation Filtration --> Cubical Persistance --> Amp, PE
    return: Array of pipelines
    """
    # Pipeline parameters
    bin_thresholds = [np.percentile(images[0], 93)/np.max(images[0])]
    directions = [np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 8)[:-1]]
    n_iterations = np.linspace(1,21, 5).astype(int).tolist()
    
    features = [('bottleneck', Amplitude(metric='bottleneck', n_jobs=-1)), 
            ('PE', PersistenceEntropy(n_jobs=-1))]

    # Make filtrations
    binned_steps = [('binarizer_{}'.format(t), Binarizer(threshold=t, n_jobs=-1)) for t in bin_thresholds]
    filtrations = [('height_{}'.format(d), HeightFiltration(direction=d, n_jobs=-1)) for d in directions]
    filtrations +=  [('erosion_{}'.format(i), ErosionFiltration(n_iterations= i, n_jobs=-1)) for i in n_iterations]
    filtrations +=  [('dilation_{}'.format(i), DilationFiltration(n_iterations= i, n_jobs=-1)) for i in n_iterations]

    # Make pipelines
    cubical_lower = ('cubical', CubicalPersistence(n_jobs=-1))

    partial_pipeline_steps = []
    partial_pipeline_steps.append([cubical_lower])
    partial_pipeline_steps.append([('inverter', Inverter(n_jobs=-1)), cubical_lower])

    for b, f in itertools.product(binned_steps, filtrations):
        partial_pipeline_steps.append([b,f, ('cubical', CubicalPersistence(n_jobs=-1))])


    feature_pipelines = []
    for s, f in itertools.product(partial_pipeline_steps, features):
        feature_pipelines.append(Pipeline(s + [f]))
        
    return feature_pipelines

def bettiCurve_pipe1(img_file):
    """
    Pipeline 1: Binarizer --> Height Filtration --> Cubical Persistance --> Betti Curve
    """
    img = cv2.imread(img_file)  
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blur the image to reduce noise
    figure_size = 9 # the dimension of the x and y axis of the kernal.
    img = cv2.blur(img,(figure_size, figure_size))
    
    shape = img.shape
    images = np.zeros((1, *shape))
    images[0] = img
    bz = Binarizer(threshold=40/255)
    binned = bz.fit_transform(images)
    p = make_pipeline(HeightFiltration(direction=np.array([1,1])), CubicalPersistence(), BettiCurve(n_bins=50))
    return p.fit_transform(binned)


def bettiCurve_pipe2(img_file):
    """
    Pipeline 2: Cubical Perisitance --> Betti Curve
    """
    img = cv2.imread(img_file)  
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blur the image to reduce noise
    figure_size = 9 # the dimension of the x and y axis of the kernal.
    # img = cv2.blur(img,(figure_size, figure_size))
    
    shape = img.shape
    images = np.zeros((1, *shape))
    images[0] = img
    p = make_pipeline(CubicalPersistence(), BettiCurve(n_bins=50))
    return p.fit_transform(images)

def persistenceEntropy(img_file):
    """
    Pipeline: Cubical Perisitance --> Persistence Entropy
    """
    img = cv2.imread(img_file)  
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blur the image to reduce noise
    figure_size = 9 # the dimension of the x and y axis of the kernal.
    img = cv2.blur(img,(figure_size, figure_size))
    
    shape = img.shape
    images = np.zeros((1, *shape))
    images[0] = img
    p = make_pipeline(CubicalPersistence(), PersistenceEntropy())
    return p.fit_transform(images)

def bettiAmplitude(img_file):
    """
    Pipeline: Cubical Perisitance --> Amplitude of Betti Curve
    """
    img = cv2.imread(img_file)  
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blur the image to reduce noise
    figure_size = 9 # the dimension of the x and y axis of the kernal.
    img = cv2.blur(img,(figure_size, figure_size))
    
    shape = img.shape
    images = np.zeros((1, *shape))
    images[0] = img
    p = make_pipeline(CubicalPersistence(), Amplitude(metric='betti'))
    return p.fit_transform(images)
