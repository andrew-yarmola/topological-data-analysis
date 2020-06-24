# Box Dimension

Estimating [box dimension](https://en.wikipedia.org/wiki/Minkowski%E2%80%93Bouligand_dimension) of underlying manifold by Monte Carlo method. Works best for uniform distributions, but produces interesting results for noise.

There is a already quite a bit of literature on dimensionality estimation: [http://www.jmlr.org/papers/volume11/mordohai10a/mordohai10a.pdf](http://www.jmlr.org/papers/volume11/mordohai10a/mordohai10a.pdf).

The most natural thing to compute is a graph representing apparent box dimension as a function of scale. Plateaus correspond the "true" dimensions of the manifold (there can be more than one, for instance consider a spiral on a torus). This has the benefit that it becomes clear how noise changes such a graph--namely, it makes the plateaus smaller and increases the overall dimension.

To determine this "true" dimension, plateau detection seems to be sufficient. A sample algorithm for plateau detection is given, but there are probably much better ones.