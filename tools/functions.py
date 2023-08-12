"""Useful functions for TSP"""
import pandas as pd
import os
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

def load(filename : str, sep : str, header = None) -> pd.DataFrame:
    """Read the file and returns a dataframe of cities"""

    datapath = './data'
    docpath = os.path.join(datapath, filename)

    df = pd.read_csv(docpath, sep = sep, header = header)
    df.set_axis(['x', 'y'], axis = 'columns', inplace = True)

    df.index = ['City_' + str(i) for i in range(len(df))]

    return df

def create(numCities : int, sideLength : int):
    """Create synthetic data for cities """

    rows_names = ['City_' + str(i) for i in range(numCities)]

    df = pd.DataFrame(np.random.randint(0, sideLength, (numCities , 2)), columns = ['x', 'y'], index = rows_names)

    return df


def computeEDM(df : pd.DataFrame) -> np.ndarray:
    """Compute the euclidean distances between cities in dataframe
    The distance bewtween City_1 and City_j is in EDM[i,j]"""

    EDM = cdist(df.values, df.values) 

    return EDM

def showCities(cities : pd.DataFrame):
    """Plot the cities"""

    fig, ax = plt.subplots()

    plt.scatter(cities['x'], cities['y'])
    plt.title('Cities in a plane')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()

def showPath(fittest : list, filename : str = None):
    """Plot the fittest path"""

    # Set dataframe in order
    d = {'x': [city.x for city in fittest], 'y' : [city.y for city in fittest]}

    # Add link to the first city
    d['x'].append(fittest[0].x)
    d['y'].append(fittest[0].y)

    df = pd.DataFrame(d)

    plt.scatter(df['x'], df['y'], zorder=1)
    plt.plot(df['x'], df['y'], zorder=2)
    plt.title('Best path in a plane')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')

    if filename is not None:
        plt.savefig(os.path.join('figures', filename), dpi=300)
    plt.show()
