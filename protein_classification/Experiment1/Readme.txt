Experiment 1:
- Look at all images with a single protein present and try to run a few topological invariant calculators on them, try to discern differences:
    a) persistence entropy
    
Todo:
    - Find 5 files for each protein type, each containing a single type of potein
    - Calculate all the different invariants

Here are the labels and what type of protein they correspond to:
0.  Nucleoplasm  
1.  Nuclear membrane   
2.  Nucleoli   
3.  Nucleoli fibrillar center   
4.  Nuclear speckles   
5.  Nuclear bodies   
6.  Endoplasmic reticulum   
7.  Golgi apparatus   
8.  Peroxisomes   
9.  Endosomes   
10.  Lysosomes   
11.  Intermediate filaments   
12.  Actin filaments   
13.  Focal adhesion sites   
14.  Microtubules   
15.  Microtubule ends   
16.  Cytokinetic bridge   
17.  Mitotic spindle   
18.  Microtubule organizing center   
19.  Centrosome   
20.  Lipid droplets   
21.  Plasma membrane   
22.  Cell junctions   
23.  Mitochondria   
24.  Aggresome   
25.  Cytosol   
26.  Cytoplasmic bodies   
27.  Rods & rings

However, our training file is missing the following:
10.  Lysosomes  
15.  Microtubule ends   
17.  Mitotic spindle   
27.  Rods & rings
Since they don't occur as the sole protein in any 1 file (27 does in a single file)

Current tasks:
+ Dump files with corresponding protein numbers into csv file
+ set up pre-processing (i.e. loading of pictures)