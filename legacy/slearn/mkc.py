# -*- coding: utf-8 -*-

"""This part is for string sequence generation"""


import numpy as np

class markovChain(object):
    def __init__(self, transition_prob):
        """
        Initialize the MarkovChain instance.
 
        Parameters
        ----------
        transition_prob: dict
            A dict object representing the transition 
            probabilities in Markov Chain. 
            Should be of the form: 
                {'state1': {'state1': 0.1, 'state2': 0.4}, 
                 'state2': {...}}
        """
        self.transition_prob = transition_prob
        self.states = list(transition_prob.keys())
 
    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time 
        instance.
 
        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        next_ = np.random.choice(self.states, p = [
              self.transition_prob[current_state][next_state] 
               for next_state in self.states]
        )
        return next_
 
    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.
 
        Parameters
        ----------
        current_state: str
            The state of the current random variable.
 
        no: int
            The number of future states to generate.
        """
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state

        return ''.join(future_states)

    
    
    
def markovMatrix(n = 26):
    '''
    Generates a Markov matrix for transition among n symbols

    Parameters
    ----------
    n: int
      number of symbols
    '''

    np.random.seed(0) # for replicability
    symbols = [chr(i+65) for i in range(n)] # list of available symbols

    markov_matrix = {}
    for i, s in enumerate(symbols):
        dim = len(symbols)

        # Probabilities of change of state (symbol)
        # Random with a decreasing probability to jump to the farthest states
        # It could be replaced by a negative logarithmic function  to make it more 
        # likely to land in the same state and change the complexity of the sequence
        probs = np.random.random(dim*3) 
        probs.sort()
        probs = np.append(probs[dim%2::2], probs[::-2])
        offset = -int(dim*1.5)
        # to keep the highest value in the current state
        probs = np.roll(probs, offset+i)[:dim] 
        probs /= probs.sum()

        markov_matrix[s] = {s: p for s, p in zip(symbols, probs)}

    return markov_matrix
