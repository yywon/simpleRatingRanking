from __future__ import division
import json
import math
import sys
import numpy as np
import cplex
from cplex.exceptions import CplexError
import time

class Evaluation:

    def __init__(self):
        self.size = 0               # Size of individual rating vector
        self.Vsub = []
        self.Ran = []               # Individual ranking vector
        self.Rat = []               # Individual rating vector
        self.Ran_from_Rat = []      # Ranking vector obtained from the rating vector, to determine inner distance
        self.seps = []              # Seperation gaps of the rating values
        self.ks = 0                 # Kemeny Snell distance between the explicit ranking vector and
                                    # the ranking vector implied by the ratings vector

    # Converts the ratings vector into its implied rankings vector for calculation of KS distance
    def calc_rankings_from_ratings(self):
        self.Ran_from_Rat = [0] * self.size

        rat_copy = np.copy(self.Rat)

        curr_rank = 1

        while sum(rat_copy) > 0:
            num_curr_rank = 0
            maxRat = max(rat_copy)
            for i in range(self.size):
                if self.Rat[i] == maxRat:
                    self.Ran_from_Rat[i] = curr_rank
                    num_curr_rank += 1
                    rat_copy[i] = 0
            curr_rank += num_curr_rank

    # Obtains the separation gaps for a ratings vector
    def calc_ratings_seps(self):
        self.seps = np.zeros((self.size, self.size), dtype=np.float)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                self.seps[i][j] = self.Rat[i] - self.Rat[j]

    # Calculates Kemeny Snell distance between the explicit ranking vector
    # and the ranking vector implied by the ratings vector
    def calc_dist_ks(self):
        for i in range(self.size):
            for j in range(i):
                self.ks += (np.abs(np.sign(self.Ran[i] - self.Ran[j])
                                   - np.sign(self.Ran_from_Rat[i] - self.Ran_from_Rat[j])))
        self.ks = 0.5 * self.ks / (self.size * (self.size-1) / 2)


class Profile:

    def __init__(self, n):
        self.num_obj = n
        self.num_jud = 0
        self.evs = []
        self.avg_KS = 0
        self.max_hops = 0
        self.sci = SCI(1, 1)
        self.groundtruth = []
        self.max_rat = 0
        self.min_rat = 0

    # Read datafile and assign ranks
    # Change this as required
    def assign_rank(self, input_file, noise):

        # Read the data file
        with open(input_file) as json_file:
            data = json.load(json_file)

        for item in data:
            given_noise = item.get('noiseLevel')
            if given_noise == noise:
                tempEval = Evaluation()
                tempEval.Ran = item.get('ranking')
                tempEval.Ran.reverse()
                tempEval.Rat = item.get('rating')
                tempEval.size = len(tempEval.Ran)
                if tempEval.size == self.num_obj:
                    tempEval.Vsub = []
                    for i in range(tempEval.size):
                        tempEval.Vsub.append(i+1)
                else:
                    tempEval.Vsub = item.get('objects ranked')
                self.groundtruth = item.get('groundtruth')
                self.max_rat = max(self.groundtruth)
                self.min_rat = min(self.groundtruth)
                self.evs.append(tempEval)
                self.num_jud += 1

    def build_Pairwise_Graph(self):
        # Initialize to all zeros
        graph = np.zeros(shape=(self.num_obj, self.num_obj), dtype=np.int)

        for e in range(len(self.evs)):
            for i in range(self.evs[e].size):
                for j in range(i + 1, self.evs[e].size):
                    graph[self.evs[e].Vsub[i] - 1][self.evs[e].Vsub[j] - 1] = 1
                    graph[self.evs[e].Vsub[j] - 1][self.evs[e].Vsub[i] - 1] = 1
        return graph

    def calc_instance_stats(self):
        # Determine the inconsistencies of each judge by counting the KS distance between
        # his/her explicit ranking vector and the ranking vector implied by his/her rating vector
        for i in range(self.num_jud):
            self.evs[i].calc_rankings_from_ratings()
            self.evs[i].calc_dist_ks()

        for i in range(self.num_jud):
            self.avg_KS += self.evs[i].ks
        self.avg_KS = round(self.avg_KS / float(self.num_jud), 2)

        # Determine the number of hops required to obtain a transitive evaluation between any
        # pair of objects
        G = self.build_Pairwise_Graph()

        self.max_hops = get_max_hops(G)

    def set_SCI(self):
        self.sci = SCI(self.num_obj, self.num_jud)

        for i in range(self.num_jud):
            # Expand reduced ranking vector into a full ranking vector (insert zeros for unranked objects)
            ran_exp = np.zeros(self.num_obj)
            for j in range(len(self.evs[i].Vsub)):
                ran_exp[self.evs[i].Vsub[j] - 1] = self.evs[i].Ran[j]
            self.sci.sm_add(ran_exp, i)

        self.sci.load_SCI()


class SCI:

    def __init__(self, n, m):
        self.num_obj = n
        self.num_jud = m
        self.mat = np.zeros((n, n), dtype=np.float)
        self.sm_array = np.zeros((self.num_jud, self.num_obj, self.num_obj), dtype=np.double)

    # Converts single ranking vectors into corresponding score matrices (based on Emonds & Mason definition)
    def ranking_to_Score_Matrix(self, a, num_ranked):
        # Initialize score matrix to all 0s
        sm_w = np.zeros((len(a), len(a)), dtype=np.double)

        for i in range(len(a)):
            if a[i] != 0:
                for j in range(len(a) - i - 1):
                    j_p = i + j + 1

                    if a[j_p] != 0:

                        # item i is strictly preferred over item j
                        if a[i] < a[j_p]:
                            sm_w[i, j_p] = 1.0 / (num_ranked * (num_ranked - 1))
                            sm_w[j_p, i] = -1.0 / (num_ranked * (num_ranked - 1))

                        # item i and j are tied
                        elif a[i] == a[j_p]:
                            sm_w[i, j_p] = 1.0 / (num_ranked * (num_ranked - 1))
                            sm_w[j_p, i] = 1.0 / (num_ranked * (num_ranked - 1))

                        # item j is strictly preferred over item i
                        else:
                            sm_w[i, j_p] = -1.0 / (num_ranked * (num_ranked - 1))
                            sm_w[j_p, i] = 1.0 / (num_ranked * (num_ranked - 1))
        return sm_w

    # Sets sm_array element idx by processing the provided ranking
    def sm_add(self, a, idx):
        num_ranked = 0
        for i in range(len(a)):
            if a[i] != 0:
                num_ranked += 1
        sm = self.ranking_to_Score_Matrix(a, num_ranked)
        self.sm_array[idx] = sm

    # Load SCI matrix; assumes sm_array has been set in full
    def load_SCI(self):
        for i in range(self.num_obj):
            for j in range(self.num_obj):
                for k in range(self.num_jud):
                    self.mat[i, j] += self.sm_array[k, i, j]


# Finds distance between nodes idx_a and idx_b using breadth first search
def find_path_length(G, idx_a, idx_b):
    stack = [(idx_a, 0)]
    done = False
    depth = 0
    distance = len(G) + 1  # This high value identifies unreachable destinations

    visited = [0] * len(G)
    visited[idx_a] = 1

    i = 0  # To traverse stack
    while not done:

        # Check all columns of current Graph row being explored
        for j in range(len(G)):
            if G[stack[i][0]][j] == 1 and visited[j] == 0:
                stack.append((j, depth + 1))  # This node belongs to the next depth level
                visited[j] = 1
                if j == idx_b:  # Found destination node
                    done = True
                    distance = depth
        i += 1  # Go to next stack index

        if i >= len(stack):
            done = True
        else:
            if stack[i][1] > depth:  # All nodes at current depth have been explored
                depth += 1  # Increase depth level

    return distance


# Determines the max number of hops required to travel between any pair of nodes in the graph
def get_max_hops(G):
    max_hops = 0

    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            hops = find_path_length(G, i, j)
            if hops > max_hops:
                max_hops = hops

    return max_hops


# Get the aggregate ranks of the objects from the values of y
def agg_rank_y(vec):
    l = len(vec)
    row_sum = [0] * l

    for i in range(l):
        sum_i = 0
        for j in range(l):
            sum_i += vec[i][j]
        row_sum[i] = 2*sum_i - 1 + l

    r = [0] * l
    numAssigned = 0
    while numAssigned < l:
        maxEntry = 0
        count = 0
        for j in range(l):
            if r[j] == 0 and row_sum[j] >= maxEntry:
                maxEntry = row_sum[j]

        for j in range(l):
            if row_sum[j] == maxEntry:
                r[j] = numAssigned + 1
                count += 1
        numAssigned += count
    return r


# Get the aggregate ranks of the objects from the values of x (for Conv_RR and RR)
def agg_rank_x(vec):
    r = [0]*len(vec)

    numAssigned = 0
    while numAssigned < len(vec):
        maxEntry = 0
        count = 0
        for j in range(len(vec)):
            if r[j] == 0 and vec[j] >= maxEntry:
                maxEntry = vec[j]

        for j in range(len(vec)):
            if vec[j] == maxEntry:
                r[j] = numAssigned + 1
                count += 1
        numAssigned += count
    return r


# KS distance between the ground truth and the aggregate ranking
def ks_GT(r1, r2):
    ks = 0
    for i in range(len(r1)):
        for j in range(i):
            ks += (np.abs(np.sign(r1[i] - r1[j])
                          - np.sign(r2[i] - r2[j])))
    return 0.5 * ks / (len(r1)*(len(r1)-1)/2)


# Get the aggregate ranks of the objects from the values of x
def g_truth(vec):
    r = [0] * len(vec)

    rat_copy = np.copy(vec)

    curr_rank = 1

    while sum(rat_copy) > 0:
        num_curr_rank = 0
        maxRat = max(rat_copy)
        for i in range(len(vec)):
            if vec[i] == maxRat:
                r[i] = curr_rank
                num_curr_rank += 1
                rat_copy[i] = 0
        curr_rank += num_curr_rank
    return r


def callibrate_model(solns, data):
    solns = np.array(solns)
    m = cplex.Cplex()
    m.objective.set_sense(m.objective.sense.minimize)

    num_judges, num_objects = np.shape(data)

    m.variables.add([0], [-cplex.infinity], [cplex.infinity], types="C", names=["cal"])

    for i in range(num_judges):
        for j in range(num_objects):
            m.variables.add([0], [-cplex.infinity], [cplex.infinity], types="C", names=["h" + str(i) + "," + str(j)])

    for i in range(num_judges):
        for j in range(num_objects):
            m.variables.add([1], [0], [cplex.infinity], types="C", names=["h_abs" + str(i) + "," + str(j)])

    for i in range(num_judges):
        for j in range(num_objects):
            if data[i, j] != 0:
                m.linear_constraints.add(
                    lin_expr=[[["h" + str(i) + "," + str(j)] + ["cal"], [1] + [-1]]],
                    senses="E", rhs=[solns[j] - data[i, j]],
                    names=["helpers" + str(i) + "," + str(j)])

    for i in range(num_judges):
        for j in range(num_objects):
            if data[i, j] != 0:
                m.linear_constraints.add(
                    lin_expr=[[["h_abs" + str(i) + "," + str(j)] + ["h" + str(i) + "," + str(j)], [1] + [-1]]],
                    senses="G", rhs=[0],
                    names=["abs_a" + str(i) + "," + str(j)])

    for i in range(num_judges):
        for j in range(num_objects):
            if data[i, j] != 0:
                m.linear_constraints.add(
                    lin_expr=[[["h_abs" + str(i) + "," + str(j)] + ["h" + str(i) + "," + str(j)], [1] + [1]]],
                    senses="G", rhs=[0],
                    names=["abs_b" + str(i) + "," + str(j)])

    m.solve()

    cal = m.solution.get_values("cal")

    for k in range(len(solns)):
        solns[k] = solns[k] + cal

    return solns


def rating_and_ranking_model(file_name, num_obj, noise_level):

    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()

    R = prof.max_rat - prof.min_rat
    L = prof.min_rat
    U = prof.max_rat
    mu = 0.01
    m = prof.num_jud
    n = prof.num_obj
    n_indiv = [0] * m               # number of objects each judge evaluates
    C = [0] * m                     # constant used in cardinal aggregation
    M = float((U - L) / mu)

    B = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            B[i][j] = round(prof.sci.mat[i][j], 5)

    for j in range(m):
        n_indiv[j] = float(len(prof.evs[j].Vsub))
        C[j] = 1 / (2 * R * math.ceil(n_indiv[j] / 2) * math.floor(n_indiv[j] / 2))

    Sep = np.zeros((m, n, n))

    for k in range(m):
        for i in range(prof.evs[k].size):
            for j in range(prof.evs[k].size):
                if j > i:
                    Sep[k][prof.evs[k].Vsub[i] - 1][prof.evs[k].Vsub[j] - 1] = prof.evs[k].seps[i][j]

    try:
        # Declare CPLEX object
        prob_RR = cplex.Cplex()

        # Declare if problem is maximization or minimization problem
        prob_RR.objective.set_sense(prob_RR.objective.sense.maximize)

        # x[i] : Decision variable
        for i in range(n):
            prob_RR.variables.add([0], [0], [math.ceil((U - L) / mu)], types="I", names=["x" + str(i + 1)])

        # y[i][j] : Decision variable
        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_RR.variables.add([2 * B[i][j]], [0], [1], types="I",
                                          names=["y" + str(i + 1) + "," + str(j + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_RR.variables.add([-2 * C[k]], [0], [cplex.infinity],
                                                  types="C",
                                                  names=["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        # Add constraints
        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_RR.linear_constraints.add(
                                lin_expr=[[["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] + ["x" + str(i + 1)]
                                           + ["x" + str(j + 1)], [1] + [-mu] + [mu]]],
                                senses="G", rhs=[-Sep[k][i][j]],
                                names=["absVal_a" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_RR.linear_constraints.add(
                                lin_expr=[[["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] + ["x" + str(i + 1)]
                                           + ["x" + str(j + 1)], [1] + [mu] + [-mu]]],
                                senses="G", rhs=[Sep[k][i][j]],
                                names=["absVal_b" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_RR.linear_constraints.add(
                        lin_expr=[[["x" + str(i + 1)] + ["x" + str(j + 1)] + ["y" + str(i + 1) + "," + str(j + 1)],
                                   [1] + [-1] + [-(M + mu)]]],
                        senses="L", rhs=[-mu],
                        names=["RatToRan_a" + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_RR.linear_constraints.add(
                        lin_expr=[[["x" + str(i + 1)] + ["x" + str(j + 1)] + ["y" + str(i + 1) + "," + str(j + 1)],
                                   [1] + [-1] + [-M]]],
                        senses="G", rhs=[-M],
                        names=["RatToRan_b" + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                if j > i:
                    prob_RR.linear_constraints.add(
                        lin_expr=[
                            [["y" + str(i + 1) + "," + str(j + 1)] + ["y" + str(j + 1) + "," + str(i + 1)], [1] + [1]]],
                        senses="G", rhs=[1],
                        names=["dispref" + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j != k != i:
                        prob_RR.linear_constraints.add(
                            lin_expr=[[["y" + str(i + 1) + "," + str(j + 1)] + ["y" + str(k + 1) + "," + str(j + 1)]
                                       + ["y" + str(i + 1) + "," + str(k + 1)], [1] + [-1] + [-1]]],
                            senses="G", rhs=[-1],
                            names=["ttvt" + str(i + 1) + "," + str(j + 1) + "," + str(k + 1)])

        start_time = time.time()
        prob_RR.solve()
        end_time = time.time()
        total_time = end_time - start_time

    except CplexError as exc:
        print(exc)
        sys.exit()

    # Used to easily calculate the aggregate ranking
    y = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                y[i][j] = round(prob_RR.solution.get_values("y" + str(i + 1) + "," + str(j + 1)))

    ranking = agg_rank_y(y)

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    x = []
    for i in range(n):
        x.append(round(prob_RR.solution.get_values("x" + str(i + 1))))

    rating = x

    # Relative optimality gap
    Gap = prob_RR.solution.MIP.get_mip_relative_gap()

    return ranking, rating, dist, Gap, total_time


def ranking_only_model(file_name, num_obj, noise_level):

    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()

    n = prof.num_obj

    B = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            B[i][j] = round(prof.sci.mat[i][j], 5)

    try:
        # Declare CPLEX object
        prob_OA = cplex.Cplex()

        # Declare if problem is maximization or minimization problem
        prob_OA.objective.set_sense(prob_OA.objective.sense.maximize)

        # Declare decision variables

        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_OA.variables.add([2 * B[i][j]], [0], [1], types="I",
                                          names=["y" + str(i + 1) + "," + str(j + 1)])

        # Add constraints

        for i in range(n):
            for j in range(n):
                if j > i:
                    prob_OA.linear_constraints.add(
                        lin_expr=[
                            [["y" + str(i + 1) + "," + str(j + 1)] + ["y" + str(j + 1) + "," + str(i + 1)], [1] + [1]]],
                        senses="G", rhs=[1],
                        names=["dispref" + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j != k != i:
                        prob_OA.linear_constraints.add(
                            lin_expr=[[["y" + str(i + 1) + "," + str(j + 1)] + ["y" + str(k + 1) + "," + str(j + 1)]
                                       + ["y" + str(i + 1) + "," + str(k + 1)], [1] + [-1] + [-1]]],
                            senses="G", rhs=[-1],
                            names=["ttvt" + str(i + 1) + "," + str(j + 1) + "," + str(k + 1)])

        start_time = time.time()
        prob_OA.solve()
        end_time = time.time()
        total_time = end_time - start_time

    except CplexError as exc:
        print(exc)
        sys.exit()

    # Used to easily calculate the aggregate ranking
    y = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                y[i][j] = round(prob_OA.solution.get_values("y" + str(i + 1) + "," + str(j + 1)))

    ranking = agg_rank_y(y)

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    # Relative optimality gap
    Gap = prob_OA.solution.MIP.get_mip_relative_gap()

    return ranking, dist, Gap, total_time


def separation_deviation_model(file_name, num_obj, noise_level, lambda1, lambda2):

    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()

    R = prof.max_rat - prof.min_rat
    L = prof.min_rat
    U = prof.max_rat
    mu = 0.01
    m = prof.num_jud
    n = prof.num_obj

    lambda_1 = lambda1  # weight of deviation cost
    lambda_2 = lambda2  # weight of seperation cost

    Sep = np.zeros((m, n, n))

    for k in range(m):
        for i in range(len(prof.evs[k].Vsub)):
            for j in range(len(prof.evs[k].Vsub)):
                if j > i:
                    Sep[k][prof.evs[k].Vsub[i] - 1][prof.evs[k].Vsub[j] - 1] = prof.evs[k].seps[i][j]

    try:
        prob_SD = cplex.Cplex()

        prob_SD.objective.set_sense(prob_SD.objective.sense.minimize)

        # x[i] : Decision variable
        for i in range(n):
            prob_SD.variables.add([0], [L / mu], [U / mu], types="I", names=["x" + str(i + 1)])

        # t[k][j] : variable for the absolute value separation cost
        for k in range(m):
            for j in range(n):
                if (j + 1) in prof.evs[k].Vsub:
                    prob_SD.variables.add([lambda_1], [-cplex.infinity], [cplex.infinity], types="C",
                                        names=["t" + str(k + 1) + "," + str(j + 1)])

        # h[k][i][j] : variable for the absolute value deviation cost
        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub:
                            if i != j:
                                prob_SD.variables.add([lambda_2], [-cplex.infinity], [cplex.infinity], types="C",
                                                    names=["h" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        # Add constraints

        # Constraints for Separation cost : G()
        for k in range(m):
            count = 0
            for j in range(n):
                if (j + 1) in prof.evs[k].Vsub:
                    prob_SD.linear_constraints.add(
                        lin_expr=[[["t" + str(k + 1) + "," + str(j + 1)] + ["x" + str(j + 1)], [1] + [-mu]]],
                        senses="G", rhs=[-prof.evs[k].Rat[count]],
                        names=["Sep_1" + str(k + 1) + "," + str(j + 1)])
                    count += 1

        for k in range(m):
            count = 0
            for j in range(n):
                if (j + 1) in prof.evs[k].Vsub:
                    prob_SD.linear_constraints.add(
                        lin_expr=[[["t" + str(k + 1) + "," + str(j + 1)] + ["x" + str(j + 1)], [1] + [mu]]],
                        senses="G", rhs=[prof.evs[k].Rat[count]],
                        names=["Sep_2" + str(k + 1) + "," + str(j + 1)])
                    count += 1

        # Constraints for Deviation cost : F()
        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub:
                            if j > i:
                                prob_SD.linear_constraints.add(
                                    lin_expr=[[["h" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] +
                                               ["x" + str(i + 1)] + ["x" + str(j + 1)],
                                               [1] + [-mu] + [mu]]],
                                    senses="G", rhs=[-Sep[k][i][j]],
                                    names=["Dev_1" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub:
                            if j > i:
                                prob_SD.linear_constraints.add(
                                    lin_expr=[[["h" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] +
                                               ["x" + str(i + 1)] + ["x" + str(j + 1)],
                                               [1] + [mu] + [-mu]]],
                                    senses="G", rhs=[Sep[k][i][j]],
                                    names=["Dev_2" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        start_time = time.time()
        prob_SD.solve()
        end_time = time.time()
        total_time = end_time - start_time

    except CplexError as exc:
        print(exc)
        sys.exit()

    # Used to easily calculate the aggregate ranking
    x = []
    for i in range(n):
        x.append(round(prob_SD.solution.get_values("x" + str(i + 1))))

    ranking = agg_rank_x(x)
    rating = x

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    # Relative optimality gap
    Gap = prob_SD.solution.MIP.get_mip_relative_gap()

    return ranking, rating, dist, Gap, total_time


def ratings_only_model(file_name, num_obj, noise_level):
    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()

    R = prof.max_rat - prof.min_rat
    L = prof.min_rat
    U = prof.max_rat
    m = prof.num_jud
    n = prof.num_obj
    n_indiv = [0] * m   # number of objects each judge evaluates
    C = [0] * m         # constant used in cardinal aggregation

    for j in range(m):
        n_indiv[j] = float(len(prof.evs[j].Vsub))
        C[j] = 1 / (2 * R * math.ceil(n_indiv[j] / 2) * math.floor(n_indiv[j] / 2))

    Sep = np.zeros((m, n, n))

    for k in range(m):
        for i in range(len(prof.evs[k].Vsub)):
            for j in range(len(prof.evs[k].Vsub)):
                if j > i:
                    Sep[k][prof.evs[k].Vsub[i] - 1][prof.evs[k].Vsub[j] - 1] = prof.evs[k].seps[i][j]

    try:
        prob_FM = cplex.Cplex()
        # Declare if problem is maximization or minimization problem
        prob_FM.objective.set_sense(prob_FM.objective.sense.minimize)

        # x[i] : Decision variable
        for i in range(n):
            prob_FM.variables.add([0], [L], [U], types="C", names=["x" + str(i + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub:
                            if j > i:
                                prob_FM.variables.add([C[k]], [0], [cplex.infinity],
                                                    types="C",
                                                    names=["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub:
                            if j > i:
                                prob_FM.linear_constraints.add(
                                    lin_expr=[
                                        [["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] + ["x" + str(i + 1)]
                                         + ["x" + str(j + 1)], [1] + [-1] + [1]]],
                                    senses="G", rhs=[-Sep[k][i][j]],
                                    names=["absVal_a" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub:
                            if j > i:
                                prob_FM.linear_constraints.add(
                                    lin_expr=[
                                        [["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] + ["x" + str(i + 1)]
                                         + ["x" + str(j + 1)], [1] + [1] + [-1]]],
                                    senses="G", rhs=[Sep[k][i][j]],
                                    names=["absVal_b" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        start_time = time.time()
        prob_FM.solve()
        end_time = time.time()
        total_time = end_time - start_time

    except CplexError as exc:
        print(exc)
        sys.exit()

    # Used to easily calculate the aggregate ranking
    x = []
    for i in range(n):
        x.append(round(prob_FM.solution.get_values("x" + str(i + 1))))

    ranking = agg_rank_x(x)

    # Define the original rating vector
    Ratings = np.zeros((m, n))
    for k in range(m):
        for i in range(n):
            Ratings[k][i] = prof.evs[k].Rat[i]
    rating = callibrate_model(x, Ratings)

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    # Relative optimality gap
    Gap = prob_FM.solution.MIP.get_mip_relative_gap()

    return ranking, rating, dist, Gap, total_time


input_file = 'responseData 11-8.json'

noise_level = 128
objects = 4

# Ratings and ranking model
RR_rank, RR_rat, RR_dist, RR_gap, RR_time = rating_and_ranking_model(input_file, objects, noise_level)
# Rankings only model
OA_rank, OA_dist, OA_gap, OA_time = ranking_only_model(input_file, objects, noise_level)
# Ratings only model (Fishbain moreno model)
CA_rank, CA_rat, CA_dist, CA_gap, CA_time = ratings_only_model(input_file, objects, noise_level)
# Separation deviation model
SD_rank, SD_rat, SD_dist, SD_gap, SD_time = separation_deviation_model(input_file, objects, noise_level, 1, 1)

print(OA_rank, OA_dist, OA_gap, OA_time)
print(RR_rank, RR_rat, RR_dist, RR_gap, RR_time)
print(CA_rank, CA_rat, CA_dist, CA_gap, CA_time)
print(SD_rank, SD_rat, SD_dist, SD_gap, SD_time)