import numpy as np

class Demand:
    '''
    This is a wrapper class for demand information
    '''
    def __init__(self, origs, dests, rates):
        self.num_commodities = len(origs)
        self.origs = origs
        self.dests = dests
        self.rates = rates
