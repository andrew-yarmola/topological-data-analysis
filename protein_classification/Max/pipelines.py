import cv2
import numpy as np
import plotly.express as px
from gtda.images import Binarizer, HeightFiltration
from gtda.pipeline import make_pipeline
from gtda.diagrams import BettiCurve
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceEntropy

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
    Pipeline 2: Cubical Perisitance --> Persistence Entropy
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