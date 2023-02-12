class City:
    def __init__(self, x : int, y : int, number : int):
        self.x = x
        self.y = y
        self.number = number

    def __repr__(self) -> str:
        return "(%s, %s)"%(str(self.x), str(self.y))

def representation(cities : list) -> list:
    '''Pass to genotypes from phenotypes (cities)'''

    chromosome = []
    n = 0

    for city in cities.iloc:
        chromosome.append(City(city.x, city.y, n))
        n += 1

    return chromosome