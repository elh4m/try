#!/usr/bin/env python
# coding: utf-8

# # HW-CI-1 Simulated Annealing - Combinatorial Optimization
# modified from https://github.com/ekoca/TSP-Simulated-Annealing
# 
# Author(s): ...
# 
# Date: ...
# 

# ## Assignement
# 
# Modify this demo:
# 
# (a) Use a different move (rearrangement) function: e.g., generate successors of a state by swapping any pair of cities in the path, rather than only adjacent cities, or reversing part of the path (e.g., reverse the BCD sequence in [ABCDE] to get [ADCBE] as the successor). What effect does the change have? 
# 
# (b) Use a different distance metric for get_value (e.g., we used the L2-norm (Euclidean distance), try the L1-norm (Manhattan distance) )
# 

# # I. Introduction
# 
# Implementation of the [simulated annealing](https://emrekoca.com/simulated-annealing/) to solve the [Traveling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem) (TSP) between US state capitals. 
# 
# 
# 
# ## Overview
# Briefly, the simulated annealing is a probablistic technique used for finding an approximate solution to an optimization problem. The TSP is an optimization problem that seeks to find the shortest path passing through every city exactly once.  In our example the TSP path is defined to start and end in the same city (so the path is a closed loop).

# In[12]:


import json
import copy
import time

import numpy as np  # contains helpful math functions like numpy.exp()
import numpy.random as random  # see numpy.random module
# import random  # alternative to numpy.random module
import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot 
import matplotlib
import matplotlib.pyplot.imread 
from keras.preprocessing.image import ImageDataGenerator
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


"""Read input data and define helper functions for visualization."""

# Map services and data available from U.S. Geological Survey, National Geospatial Program.
# Please go to http://www.usgs.gov/visual-id/credit_usgs.html for further information
map = matplotlib.pyplot.imread  (r'C:\Users\Asr\Desktop\map.png')  # US States & Capitals map

# List of 30 US state capitals and corresponding coordinates on the map
with open('capitals.json', 'r') as capitals_file:
    capitals = json.load(capitals_file)
capitals_list = list(capitals.items())
def prepare_image(file):
    img_path = (r'C:\Users\Asr\Desktop\map.png')
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return 
    keras.applications.mobilenet.preprocess_imput(img_array_expanded_dims)

preprocessed_image = prepare_image(r"C:\Users\Asr\Desktop\map.png")
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
results

def show_path(path, starting_city, w=12, h=8):
    """Plot a TSP path overlaid on a map of the US States & their capitals."""
    x, y = list(zip(*path))
    _, (x0, y0) = starting_city
    plt.imshow(map)
    plt.plot(x0, y0, 'y*', markersize=15)  # y* = yellow star for starting point
    plt.plot(x + x[:1], y + y[:1])  # include the starting point at the end of path
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


# ## II. Simulated Annealing -- Main Loop
# 
# The main loop of simulated annealing repeatedly generates successors in the neighborhood of the current state and considers moving there according to an acceptance probability distribution parameterized by a cooling schedule.  See the [simulated-annealing function](https://github.com/aimacode/aima-pseudocode/blob/master/md/Simulated-Annealing.md) pseudocode from the AIMA textbook online at github.

# In[3]:


def simulated_annealing(problem, schedule):
    """The simulated annealing algorithm, a version of stochastic hill climbing
    where some downhill moves are allowed. Downhill moves are accepted readily
    early in the annealing schedule and then less often as time goes on. The
    schedule input determines the value of the temperature T as a function of
    time.
    
    Parameters
    ----------
    problem : Problem
        An optimization problem, already initialized to a random starting state.
        The Problem class interface must implement a callable method
        "successors()" which returns states in the neighborhood of the current
        state, and a callable function "get_value()" which returns a fitness
        score for the state. (See the `TravelingSalesmanProblem` class below
        for details.)

    schedule : callable
        A function mapping time to "temperature". "Time" is equivalent in this
        case to the number of loop iterations.
    
    Returns
    -------
    Problem
        An approximate solution state of the optimization problem
    """     
    current = problem
    t = 1
    while True:
        T = schedule(t)
        if T == 0:
            return current
        next = random.choice(current.successors(), 1)[0]
        delta = next.get_value() - current.get_value()
        if delta > 0 or random.random_sample() < np.e**(delta/T): 
            current = next
        t += 1


# ## III. Representing the Problem
# 
# In order to use simulated annealing we need to build a representation of the problem domain.  The choice of representation can have a significant impact on the performance of simulated annealing and other optimization techniques.  Since the TSP deals with a close loop that visits each city in a list once, we will represent each city by a tuple containing the city name and its position specified by an (x,y) location on a grid.  The _state_ will then consist of an ordered sequence (a list) of the cities; the path is defined as the sequence generated by traveling from each city in the list to the next in order.

# In[4]:


class TravelingSalesmanProblem:
    """Representation of a traveling salesman optimization problem.  The goal
    is to find the shortest path that visits every city in a closed loop path.
    
    Parameters
    ----------
    cities : list
        A list of cities specified by a tuple containing the name and the x, y
        location of the city on a grid. e.g., ("Atlanta", (585.6, 376.8))
    
    Attributes
    ----------
    names
    coords
    path : list
        The current path between cities as specified by the order of the city
        tuples in the list.
    """
    def __init__(self, cities):
        self.path = copy.deepcopy(cities)
    
    def copy(self):
        """Return a copy of the current board state."""
        new_tsp = TravelingSalesmanProblem(self.path)
        return new_tsp
    
    @property
    def names(self):
        """Strip and return only the city name from each element of the
        path list. For example,
            [("Atlanta", (585.6, 376.8)), ...] -> ["Atlanta", ...]
        """
        names, _ = zip(*self.path)
        return names
    
    @property
    def coords(self):
        """Strip the city name from each element of the path list and return
        a list of tuples containing only pairs of xy coordinates for the
        cities. For example,
            [("Atlanta", (585.6, 376.8)), ...] -> [(585.6, 376.8), ...]
        """
        _, coords = zip(*self.path)
        return coords
    
    def successors(self):
        """Return a list of states in the neighborhood of the current state.
        In general, a path of N cities will have N neighbors (note that path wraps
        around the end of the list between the first and last cities).
        
        I have implemented two different successors functions to see performance
        and what effect does the change have?

        Returns
        -------
        list<Problem>
            A list of TravelingSalesmanProblem instances initialized with their list
            of cities set to one of the neighboring permutations of cities in the
            present state
        """
        def _successors_permutations(L):
            """Experimental successor function: Return a list of states in the 
            neighborhood of the current state by taking a permutation on current 
            state so a path of N cities will have N! neighbors.
            """
            import itertools
            states = []
            paths = list(itertools.permutations(self.path))
            for path in paths:
                states.append(TravelingSalesmanProblem(path))
            return states
        
        def _successors_neighbors(L):
            """Default successor function: Return a list of states in the 
            neighborhood of the current state by switching the order in which 
            any adjacent pair of cities is visited.
            
            For example, if the current list of cities (i.e., the path) is [A, B, C, D]
            then the neighbors will include [A, B, D, C], [A, C, B, D], [B, A, C, D],
            and [D, B, C, A]. (The order of successors does not matter.)
            """
            states = [TravelingSalesmanProblem(self.path[-1:]+self.path[1:-1]+self.path[:1])]
            for i in range(L-1):
                states.append(TravelingSalesmanProblem(self.path[:i] + self.path[i:(i+2)][::-1] + self.path[(i+2):]))
            return states
        
        L = len(self.path)
        return _successors_neighbors(L)


    def get_value(self):
        """Calculate the total length of the closed-circuit path of the current
        state by summing the distance between every pair of adjacent cities.  Since
        the default simulated annealing algorithm seeks to maximize the objective
        function, return -1x the path length. (Multiplying by -1 makes the smallest
        path the smallest negative number, which is the maximum value.)
        
        Use a different distance metric for get_value (e.g., we used the L2-norm (Euclidean distance),
        try the L1-norm (manhattan distance) or L ∞∞ -norm (uniform norm)
        
        Returns
        -------
        float
            A floating point value with the total cost of the path given by visiting
            the cities in the order according to the self.cities list
        
        Notes
        -----
            (1) Remember to include the edge from the last city back to the
            first city
            
            (2) Remember to multiply the path length by -1 so that simulated
            annealing finds the shortest path
        """
        def _distance(p1, p2):
            return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
            
        L = len(self.path)
        return -1 * sum([_distance(self.path[i][1], self.path[(i+1)%L][1]) for i in range(L)])


# In[5]:


# Test case 1: Construct an instance of the TravelingSalesmanProblem
test_cities = [('DC', (11, 1)), ('SF', (0, 0)), ('PHX', (2, -3)), ('LA', (0, -4))]
tsp = TravelingSalesmanProblem(test_cities)
assert(tsp.path == test_cities)


# ### Testing TravelingSalesmanProblem
# The following tests should validate the class constructor and functionality of the `successors()` and `get_value()` methods.

# In[6]:


# Test case 2: Test the successors() method -- no output means the test passed
successor_paths = [x.path for x in tsp.successors()]
#[print (x) for x in successor_paths]
assert((x in [[('LA', (0, -4)), ('SF', (0, 0)), ('PHX', (2, -3)), ('DC', (11, 1))],
                 [('SF', (0, 0)), ('DC', (11, 1)), ('PHX', (2, -3)), ('LA', (0, -4))],
                 [('DC', (11, 1)), ('PHX', (2, -3)), ('SF', (0, 0)), ('LA', (0, -4))],
                 [('DC', (11, 1)), ('SF', (0, 0)), ('LA', (0, -4)), ('PHX', (2, -3))]]
          for x in successor_paths))
# uncomment this for _successors_neighbors
#assert(all(x in [[('LA', (0, -4)), ('SF', (0, 0)), ('PHX', (2, -3)), ('DC', (11, 1))],
#                 [('SF', (0, 0)), ('DC', (11, 1)), ('PHX', (2, -3)), ('LA', (0, -4))],
#                 [('DC', (11, 1)), ('PHX', (2, -3)), ('SF', (0, 0)), ('LA', (0, -4))],
#                 [('DC', (11, 1)), ('SF', (0, 0)), ('LA', (0, -4)), ('PHX', (2, -3))]]
#          for x in successor_paths))


# In[7]:


# Test case 3: Test the get_value() method -- no output means the test passed
assert(np.allclose(tsp.get_value(), -28.97, atol=1e-3))


# ## IV. Define the Temperature Schedule
# 
# The most common temperature schedule is simple exponential decay:
# $T(t) = \alpha^t T_0$
# 
# (Note that this is equivalent to the incremental form $T_{i+1} = \alpha T_i$, but implementing that form is slightly more complicated because you need to preserve state between calls.)
# 
# In most cases, the valid range for temperature $T_0$ can be very high (e.g., 1e8 or higher), and the _decay parameter_ $\alpha$ should be close to, but less than 1.0 (e.g., 0.95 or 0.99).  Think about the ways these parameters effect the simulated annealing function.  Try experimenting with both parameters to see how it changes runtime and the quality of solutions.
# 
# You can also experiment with other schedule functions -- linear, quadratic, etc.  Think about the ways that changing the form of the temperature schedule changes the behavior and results of the simulated annealing function.

# In[8]:


alpha = 0.95
temperature=1e4

def schedule(time):
    return alpha**time * temperature


# ### Testing the Temperature Schedule
# The following tests should validate the temperature schedule function and perform a simple test of the simulated annealing function to solve a small TSP test case

# In[9]:


# Test case 4: Test the schedule() function -- no output means that the tests passed
assert(np.allclose(alpha, 0.95, atol=1e-3))
assert(np.allclose(schedule(0), temperature, atol=1e-3))
assert(np.allclose(schedule(10), 5987.3694, atol=1e-3))


# In[10]:


# Test case 5: Failure implies that the initial path of the test case has been changed
assert(tsp.path == [('DC', (11, 1)), ('SF', (0, 0)), ('PHX', (2, -3)), ('LA', (0, -4))])
result = simulated_annealing(tsp, schedule)
print("Initial score: {}\nStarting Path: {!s}".format(tsp.get_value(), tsp.path))
print("Final score: {}\nFinal Path: {!s}".format(result.get_value(), result.path))
assert(tsp.path != result.path)
assert(result.get_value() > tsp.get_value())


# ## V. Run Simulated Annealing on a Larger TSP
# Now we are ready to solve a TSP on a bigger problem instance by finding a shortest-path circuit through several of the US state capitals.
# 
# You can increase the `num_cities` parameter up to 30 to experiment with increasingly larger domains. 
# 
# Try running the solver repeatedly -- how stable are the results?

# In[11]:


# Create the problem instance and plot the initial state
num_cities = 30
capitals_tsp = TravelingSalesmanProblem(capitals_list[:num_cities])
starting_city = capitals_list[0]
print("Initial path value: {:.2f}".format(-capitals_tsp.get_value()))
print(capitals_list[:num_cities])  # The start/end point is indicated with a yellow star
show_path(capitals_tsp.coords, starting_city)


# In[12]:


# set the decay rate and initial temperature parameters, then run simulated annealing to solve the TSP
alpha = 0.95 # try .99
temperature=1e6
t_before = time.time()  
result = simulated_annealing(capitals_tsp, schedule)
print("Final path length: {:.2f}".format(-result.get_value()), end=" ") 
print("  Time = ", round (time.time()-t_before, 2), " seconds")
print(result.path)
show_path(result.coords, starting_city)


# # Additional experiments (optional)
# 
# Change the number of cities in the final map (between 10 and 30). How are your results affected? Why?
# 
# Change the alpha and temperature parameters. How do they affect the results? 
# 
# Use a different schedule function (something other than exponential decay). Is the algorithm still effective? 
# 
