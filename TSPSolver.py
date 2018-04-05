#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import pickle


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None
        self._time_elapsed = 0
        self._nsubprobs = 0
        self._npruned = 0
        self._bssf = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour
        </summary>
        <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (
not counting initial BSSF estimate)</returns> '''

    def defaultRandomTour(self, start_time, time_allowance=60.0):

        results = {}

        start_time = time.time()

        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        while not foundTour:
            # create a random permutation
            perm = np.random.permutation(ncities)

            # for i in range( ncities ):
            # swap = i
            # while swap == i:
            # swap = np.random.randint(ncities)
            # temp = perm[i]
            # perm[i] = perm[swap]
            # perm[swap] = temp

            route = []

            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])

            bssf = TSPSolution(route)
            # bssf_cost = bssf.cost()
            # count++;
            count += 1

            # if costOfBssf() < float('inf'):
            if bssf.costOfRoute() < np.inf:
                # Found a valid route
                foundTour = True
        #} while (costOfBssf() == double.PositiveInfinity);                // until a valid route is found
        # timer.Stop();

        # costOfBssf().ToString();                          // load results array
        results['cost'] = bssf.costOfRoute()
        results['time'] = time.time() - start_time
        results['count'] = count
        results['soln'] = bssf

       # return results;
        return results

    # O(|cities|)
    def greedy(self, start_time, time_allowance=60.0):
        # O(|cities|log(|cities|))
        cities = sorted(self._scenario.getCities(), key=lambda x: x._index)
        # O(|cities|^2)
        cost_matrix = self.get_cost_matrix(cities)

        # O(|cities|^2)
        min_indexes = np.argmin(cost_matrix, axis=1)
        bssf = TSPSolution(cities)

        for city_index in range(len(cities)):
            current = city_index
            route = []

            # O(|cities|)
            while len(route) < len(cities):
                route.append(cities[current])
                cost_matrix[:, current] = np.inf
                current = min_indexes[current]
                min_indexes = np.argmin(cost_matrix, axis=1)

            sol = TSPSolution(route)
            if sol.costOfRoute() < bssf.costOfRoute():
                bssf = TSPSolution(route)  # O(|cities|)

        results = {}
        results['cost'] = bssf.costOfRoute()
        results['time'] = time.time() - start_time
        results['count'] = 0
        results['soln'] = bssf

        return results

    # O(|cities|)
    def reduce_costs(self, M):
        total_reduction = 0

        # # Rows
        # row_mins = M[np.arange(M.shape[1]), np.argmin(M, axis=1)]
        # sub_matrix = np.repeat(row_mins, M.shape[1]).reshape(M.shape)
        # sub_matrix[sub_matrix == np.inf] = 0
        # M -= sub_matrix

        # Rows O(|cities|) if we consider asignemnt and
        # argmin as O(1)
        mins = np.argmin(M, axis=1)
        for i in range(M.shape[1]):
            val = M[i, mins[i]]
            if val != np.inf:
                total_reduction += val
                M[i, :] -= val

        M = M.T

        mins = np.argmin(M, axis=1)
        for i in range(M.shape[1]):
            val = M[i, mins[i]]
            if val != np.inf:
                total_reduction += val
                M[i, :] -= val

        # # Columns same as Rows
        # for i, index in enumerate(np.argmin(M, axis=0)):
        #     val = M[index, i]
        #     if val != np.inf:
        #         total_reduction += val
        #         M[:, i] -= val

        return M.T, total_reduction

    # O(|cities|)
    def expand_subprobs(self, parent, cities, cost_matrix, costOfRoute):
        substates = []
        pindex = parent.city._index
        rcm = parent.rcm

        # O(|cities|)
        for j in range(rcm.shape[1]):
            if rcm[pindex, j] == np.inf or cost_matrix[pindex, j] + parent.lb > costOfRoute:
                continue

            child = State()

            # O(1) or O(|cities|^2) depending on if we count
            # vectorized code
            child.rcm = rcm.copy()

            child.lb = parent.lb
            child.lb += rcm[pindex, j]

            # Set row, column, and back edge to infinity
            child.rcm[pindex, :] = np.inf
            child.rcm[:, j] = np.inf
            child.rcm[j, pindex] = np.inf

            # Reduce
            # O(|cities|) for self.reduce_costs
            child.rcm, total_reduction = self.reduce_costs(child.rcm)
            child.lb += total_reduction

            if child.lb >= costOfRoute:
                self._npruned += 1
                continue

            child.city = cities[j]

            # O(1) for costTo
            child.path_to = parent.path_to.copy()  # O(|cities|)
            child.path_to.append(parent.city)  # O(1)

            substates.append(child)  # O(1)

        return substates

    # O(|cities|^2)
    def get_cost_matrix(self, cities):
        ncities = len(cities)
        M = np.zeros((ncities, ncities))
        for i, a in enumerate(cities):
            for j, b in enumerate(cities):
                M[i, j] = a.costTo(b)  # O(1)

        return M

    # O(1)
    def get_priority(self, state):
        return state.lb / len(state.path_to)
        # return state.lb - 200 * len(state.path_to)

    def branchAndBound(self, start_time, time_allowance=60.0):
        # with open('solver', 'wb') as f:
        #     pickle.dump(self, f)
        # sorting is O(|cities|log|cities|)
        cities = sorted(self._scenario.getCities(), key=lambda x: x._index)
        # cost matrix costs O(|cities|^2)
        cost_matrix = self.get_cost_matrix(cities)
        count = 0                                   # Number of times BSSF updated
        # O(|cities|) for greedy
        self._bssf = self.greedy(time.time())['soln']

        self._nsubprobs = 0
        self._npruned = 0
        # O(|cities|) for reduce_costs
        rcm, _ = self.reduce_costs(cost_matrix.copy())

        first_state = State(
            rcm,
            0,
            0,
            None,
            cities[0],
            []
        )  # O(1)

        q = [(1, 0, first_state)]
        max_q = 0
        entry_count = 1
        costOfRoute = self._bssf.costOfRoute()
        # Worst case loop iterations = O(|states||cities|) if every expansion
        # expanded to substates for all cities
        while len(q) != 0 and time.time() - start_time < time_allowance:
            if len(q) > max_q:
                max_q = len(q)

            _, _, state = heapq.heappop(q)  # O(log|states|)
            if state.lb > costOfRoute:  # O(|cities|)
                self._npruned += 1
            else:
                # Worst case, number of substates = O(|cities|)
                substates = self.expand_subprobs(
                    state, cities, cost_matrix, costOfRoute)
                self._nsubprobs += len(substates)
                # Worst case, loop iterations = O(|cities|)
                for substate in substates:
                    is_solution = False

                    # O(|cities|^2) for min calculation
                    if np.min(substate.rcm) == np.inf:
                        if (len(substate.path_to) == len(cities)):
                            is_solution = True
                        else:
                            self._npruned += 1
                            continue

                    if is_solution:
                        # print('solution found', list(map(
                        #     lambda x: x._name, substate.path_to)))
                        possible_solution = TSPSolution(substate.path_to)
                        possible_solution_cost = possible_solution.costOfRoute()
                        # O(|cities|)
                        if possible_solution_cost < costOfRoute:
                            self._bssf = possible_solution  # O(|cities|)
                            count += 1
                            costOfRoute = possible_solution_cost

                    elif substate.lb < costOfRoute:  # O(|cities|)
                        # O(log|states|)
                        heapq.heappush(
                            q, (self.get_priority(substate), entry_count, substate))
                        entry_count += 1

                    else:
                        self._npruned += 1

        time_taken = time.time() - start_time
        print('time_taken: ', time_taken)
        print('maximum q size', max_q)
        print('cost', costOfRoute)
        print('self._bssf updates', count)
        print('states pruned', self._npruned)
        print('total states', self._nsubprobs)
        print('========')

        results = {}
        results['cost'] = costOfRoute  # O(|cities|)
        results['time'] = time_taken
        results['count'] = count
        results['soln'] = self._bssf

        return results

    def fancy(self, start_time, time_allowance=60.0):
        pass
