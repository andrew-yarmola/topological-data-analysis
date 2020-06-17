# Orbit visualizer

Solar system orbital data (called an "ephemeris") can be found on [this website](http://astropixels.com/ephemeris/ephemeris.html)

Each page gives observations of the orbit of some planet over the course of a year in the form of a 14x365 table. The most important columns are "RA" (horizontal coordinate in the sky) and "dec" (vertical coordinate in the sky).

This table should be copied into a text document, as in the sample "Mercury 2020.txt". The file Parse.py can be run from the command line to convert the relevant data to a csv, e.g. "Mercury 2020.csv", which is then used for the visualization.

The notebook provides a tutorial for the visualization.