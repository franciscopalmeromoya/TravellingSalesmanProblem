# A Genetic Algorithm for the Traveling Salesman Problem
## Overview
The present repository shows a Genetic Algorithm approach for the Traveling Salesman Problem. The study primarily
investigates the interplay between crossover probability and the effectiveness (including efficiency) of the genetic
algorithm. It delves into how the crossover probability influences the quality and convergence speed of results,
demonstrated through instances of varying complexities in the Traveling Salesman Problem.
## Implementation
The code designed for this problem is developed from scratch. It is written in Python and make use of common libraries of the language such as ``numpy, scipy, matplotlib, etc``. The code structure is divided into two main package, 
one for the GA and other for usuful functions for the specific problem.
The GeneticAlgorithm package is based on ``class GeneticAlgorithm``, a Python class which handles the main
functions of any GA. However, since it is developed for a concrete problem, namely the TSP, the current implemented
function are specific for the problem at hand. The main advantage of the code design is its modular approach, making
possible to reuse the code recoursively. If one want to use a different operator, it only requires the development of the
new operator, and then insert it into the code.
Other useful functions specific for the TSP are developed into the ``tools`` package. There,we can find plot functions
for the path and cities representation into a 2-dimensional plane or functions to load the cities coordinates from a file.
