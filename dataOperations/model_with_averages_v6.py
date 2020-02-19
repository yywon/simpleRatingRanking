from __future__ import division
import json
import math
import sys
import numpy as np
import cplex
from cplex.exceptions import CplexError
import time
import matplotlib.pyplot as plt
import scipy.spatial
from scipy.spatial.distance import cdist
import scipy.stats as ss


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
        #self.ks = 0.5 * self.ks / (self.size * (self.size-1) / 2)
        self.ks = 0.5 * self.ks

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
        self.min_rat = 10000

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

# Option 2. (Comment 107, 108, Uncomment 114-119)
# Tried to set the max and the min value for the range as the max and min value in the profile.
# However, the max and min value are too deviated.
#                if max(tempEval.Rat) > self.max_rat:
#                    self.max_rat = max(tempEval.Rat)                    
#                if min(tempEval.Rat) < self.min_rat:
#                    self.min_rat = min(tempEval.Rat)

                self.evs.append(tempEval)
                self.num_jud += 1
                
# Option 3. (Uncomment 109, 110, 128, 129) => no solution....... why??????????
# With Line 111-112, tried to set the max and the min value for the range
# as the average of max value and that of min value               

#        self.max_rat = self.max_rat/self.num_jud
#        self.min_rat = self.min_rat/self.num_jud


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
        # majority margin 
        self.maximin = np.zeros((n, n), dtype=np.float)
        self.mm_array = np.zeros((self.num_jud, self.num_obj, self.num_obj), dtype=np.double)

    # Converts single ranking vectors into corresponding score matrices (based on Emonds & Mason definition)
    def ranking_to_Score_Matrix(self, a, num_ranked):
        # Initialize score matrix to all 0s
        sm_w = np.zeros((len(a), len(a)), dtype=np.double)
        mm_w = np.zeros((len(a), len(a)), dtype=np.double)

        for i in range(len(a)):
            if a[i] != 0:
                for j in range(len(a) - i - 1):
                    j_p = i + j + 1

                    if a[j_p] != 0:

                        # item i is strictly preferred over item j
                        if a[i] < a[j_p]:
                            sm_w[i, j_p] = 1.0 / (num_ranked * (num_ranked - 1))
                            sm_w[j_p, i] = -1.0 / (num_ranked * (num_ranked - 1))
                            mm_w[i, j_p] = 1.0

                        # item i and j are tied
                        elif a[i] == a[j_p]:
                            sm_w[i, j_p] = 1.0 / (num_ranked * (num_ranked - 1))
                            sm_w[j_p, i] = 1.0 / (num_ranked * (num_ranked - 1))

                        # item j is strictly preferred over item i
                        else:
                            sm_w[i, j_p] = -1.0 / (num_ranked * (num_ranked - 1))
                            sm_w[j_p, i] = 1.0 / (num_ranked * (num_ranked - 1))
                            mm_w[j_p, i] = 1.0
        return sm_w, mm_w


    # Sets sm_array element idx by processing the provided ranking
    def sm_add(self, a, idx):
        num_ranked = 0
        for i in range(len(a)):
            if a[i] != 0:
                num_ranked += 1
        sm, mm = self.ranking_to_Score_Matrix(a, num_ranked)
        self.sm_array[idx] = sm
        self.mm_array[idx] = mm

    # Load SCI matrix; assumes sm_array has been set in full
    def load_SCI(self):
        for i in range(self.num_obj):
            for j in range(self.num_obj):
                for k in range(self.num_jud):
                    self.mat[i, j] += self.sm_array[k, i, j]
                    self.maximin[i, j] += self.mm_array[k, i, j]

# aggregated ranking from maximin rule
def maximin(mat):
    
    min_vec = [0] * len(mat)
    agg_rank = [0] * len(mat)
    
    for i in range(len(mat)):
        min_vec[i] = min(i for i in mat[i] if i > 0)   
    agg_rank = ss.rankdata([-1 * i for i in min_vec]).astype(int)
    
    return agg_rank

# aggregated ranking from copeland rule (copeland score: |x>y|-|y>x|)
def copeland(mat):
    
    cpld_score = [0] * len(mat)  
    
    for i in range(len(mat)):
        for j in range(len(mat)):
            if (mat[i,j] - mat[j,i]) > 0:
                cpld_score[i] += 1
                       
    agg_rank = ss.rankdata([-1 * i for i in cpld_score]).astype(int)
             
    return agg_rank
    
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
    #return 0.5 * ks / (len(r1)*(len(r1)-1)/2)
    return 0.5 * ks

    
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

# Calculate all the parameters
def calculate_parameters(p):
    
    prof = p
    num_obj = prof.num_obj
    num_jud = prof.num_jud
    B = np.zeros((num_obj, num_obj))
    L = prof.min_rat
    U = prof.max_rat
    R = U - L
    n_indiv = [0] * num_jud
    C = [0] * num_jud
    Sep = np.zeros((num_jud, num_obj, num_obj))

    for i in range(num_obj):
        for j in range(num_obj):
            B[i][j] = round(prof.sci.mat[i][j], 5)

    for j in range(num_jud):
        n_indiv[j] = float(len(prof.evs[j].Vsub))
        C[j] = 1 / (4 * R * math.ceil(n_indiv[j] / 2) * math.floor(n_indiv[j] / 2))

    for k in range(num_jud):
        for i in range(prof.evs[k].size):
            for j in range(prof.evs[k].size):
                if j > i:
                    Sep[k][prof.evs[k].Vsub[i] - 1][prof.evs[k].Vsub[j] - 1] = prof.evs[k].seps[i][j]

    # input rating matrix
    rat_matrix = np.zeros((num_jud, num_obj))
    for k in range(num_jud):
        for i in range(num_obj):
            rat_matrix[k][i] = prof.evs[k].Rat[i]

    return B, C, Sep, rat_matrix

# calibration model for rating values
def calibrate_model(solns, data):

    # this is the solution vector (aggregate rating vector)    
    solns = np.array(solns)
    
    m = cplex.Cplex()
    m.objective.set_sense(m.objective.sense.minimize)

    num_judges, num_objects = np.shape(data)

    # add the variable "c" to the model
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

    m.set_log_stream(None)
    m.set_error_stream(None)
    m.set_warning_stream(None)
    m.set_results_stream(None)
    
    m.solve()

    cal = m.solution.get_values("cal")

    for k in range(len(solns)):
        solns[k] = solns[k] + cal

    return solns

def rating_and_ranking_model(file_name, num_obj, noise_level, lambda_rat, lambda_ran):

#    print("rating and ranking only")
    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()

    B, C, Sep, input_ratings = calculate_parameters(prof)
    
    n = prof.num_obj
    m = prof.num_jud
    L = prof.min_rat
    U = prof.max_rat

    mu = 1.0
    M = float((U - L) / mu)

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
                    prob_RR.variables.add([lambda_ran * B[i][j]], [0], [1], types="I",
                                          names=["y" + str(i + 1) + "," + str(j + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_RR.variables.add([-4 * lambda_rat * C[k]], [0], [cplex.infinity],
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

        # Following two equations are modified for image ranking problem
        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_RR.linear_constraints.add(
                        lin_expr=[[["x" + str(i + 1)] + ["x" + str(j + 1)] + ["y" + str(i + 1) + "," + str(j + 1)],
                                   [-1] + [1] + [-(M + mu)]]],
                        senses="L", rhs=[-mu],
                        names=["RatToRan_a" + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_RR.linear_constraints.add(
                        lin_expr=[[["x" + str(i + 1)] + ["x" + str(j + 1)] + ["y" + str(i + 1) + "," + str(j + 1)],
                                   [-1] + [1] + [-M]]],
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
        
        prob_RR.set_log_stream(None)
        prob_RR.set_error_stream(None)
        prob_RR.set_warning_stream(None)
        prob_RR.set_results_stream(None)
        
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
        x.append(round(mu*prob_RR.solution.get_values("x" + str(i + 1)) + L))
    # calibrate the rating values
    rating = calibrate_model(x, input_ratings)

    # Relative optimality gap
    Gap = prob_RR.solution.MIP.get_mip_relative_gap()

    #get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating,true_rating)
    L2 = scipy.spatial.distance.euclidean(rating,true_rating)/(max(true_rating)-min(true_rating))
    
#    L2 = sum([i**2 for i in (rating-true_rating)/(max(true_rating)-min(true_rating))])
#    L2 = scipy.spatial.distance.euclidean(rating,true_rating)/R
#    L2 = sum([i**2 for i in (rating-true_rating)/R])

    return ranking, rating, dist, Gap, total_time, L1, L2


def ranking_only_model(file_name, num_obj, noise_level):

#    print("ranking only")

    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()

    B, C, Sep, input_ratings = calculate_parameters(prof)

    n = prof.num_obj

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
        
        prob_OA.set_log_stream(None)
        prob_OA.set_error_stream(None)
        prob_OA.set_warning_stream(None)
        prob_OA.set_results_stream(None)
        
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

#    print("sep and dev")
    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()

    B, C, Sep, input_ratings = calculate_parameters(prof)

    n = prof.num_obj
    m = prof.num_jud
    L = prof.min_rat
    U = prof.max_rat

    lambda_1 = lambda1  # weight of deviation cost
    lambda_2 = lambda2  # weight of seperation cost
    
    try:
        prob_SD = cplex.Cplex()

        prob_SD.objective.set_sense(prob_SD.objective.sense.minimize)

        # x[i] : Decision variable
        for i in range(n):
            prob_SD.variables.add([0], [L], [U], types="I", names=["x" + str(i + 1)])

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
                            if j > i:
                                prob_SD.variables.add([lambda_2], [-cplex.infinity], [cplex.infinity], types="C",
                                                    names=["h" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        # Add constraints

        # Constraints for Separation cost : G()
        for k in range(m):
            count = 0
            for j in range(n):
                if (j + 1) in prof.evs[k].Vsub:
                    prob_SD.linear_constraints.add(
                        lin_expr=[[["t" + str(k + 1) + "," + str(j + 1)] + ["x" + str(j + 1)], [1] + [-1]]],
                        senses="G", rhs=[-prof.evs[k].Rat[count]],
                        names=["Sep_1" + str(k + 1) + "," + str(j + 1)])
                    count += 1

        for k in range(m):
            count = 0
            for j in range(n):
                if (j + 1) in prof.evs[k].Vsub:
                    prob_SD.linear_constraints.add(
                        lin_expr=[[["t" + str(k + 1) + "," + str(j + 1)] + ["x" + str(j + 1)], [1] + [1]]],
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
                                               ["x" + str(i + 1)] + ["x" + str(j + 1)], [1] + [-1] + [1]]],
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
                                               ["x" + str(i + 1)] + ["x" + str(j + 1)], [1] + [1] + [-1]]],
                                    senses="G", rhs=[Sep[k][i][j]],
                                    names=["Dev_2" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        start_time = time.time()
        
        prob_SD.set_log_stream(None)
        prob_SD.set_error_stream(None)
        prob_SD.set_warning_stream(None)
        prob_SD.set_results_stream(None)
        
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

    # calibrate the data
    rating = calibrate_model(x, input_ratings)

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    # Relative optimality gap
    Gap = prob_SD.solution.MIP.get_mip_relative_gap()

    #get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating,true_rating)
    # distance normalized by the maximum and minimum element of ground truth
    L2 = scipy.spatial.distance.euclidean(rating,true_rating)/(max(true_rating)-min(true_rating))
#    L2 = sum([i**2 for i in (rating-true_rating)/(max(true_rating)-min(true_rating))])

    # distance normalized by the maximum and minimum element of the profile
#    L2 = scipy.spatial.distance.euclidean(rating,true_rating)/R
#    L2 = sum([i**2 for i in (rating-true_rating)/R])

    return ranking, rating, dist, Gap, total_time, L1, L2


# ratings and ranking model with spearman footrule distance and NPCK distance
def SF_ratings_and_ranking_model(file_name, num_obj, noise_level, lambda_rat, lambda_ran):

#    print("SF ratings and ranking")
    
    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()

    B, C, Sep, input_ratings = calculate_parameters(prof)

    n = prof.num_obj
    m = prof.num_jud
    L = prof.min_rat
    U = prof.max_rat
    mu = 1.0
    M = float((U - L) / mu)

    try:
        # Declare CPLEX object
        prob_SF = cplex.Cplex()

        # Declare if problem is maximization or minimization problem
        prob_SF.objective.set_sense(prob_SF.objective.sense.minimize)

        # y[i][j] : Decision variable
        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_SF.variables.add([0], [0], [1], types="I", names=["y" + str(i + 1) + "," + str(j + 1)])

        # h[k][i]: additional variable for ranking part
        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    prob_SF.variables.add([lambda_ran], [-cplex.infinity], [cplex.infinity],
                                            types="C", names=["h" + str(k + 1) + "," + str(i + 1)])

        # x[i] : Decision variable
        for i in range(n):
            prob_SF.variables.add([0], [0], [math.ceil((U - L) / mu)], types="I", names=["x" + str(i + 1)])

        # t[k][i][j]: additional variable for rating part
        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_SF.variables.add([4 * lambda_rat * C[k]], [0], [cplex.infinity],
                                                  types="C",
                                                  names=["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        # Add constraints
        # Constraints for rating
        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_SF.linear_constraints.add(
                                lin_expr=[[["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] + ["x" + str(i + 1)]
                                           + ["x" + str(j + 1)], [1] + [-mu] + [mu]]],
                                senses="G", rhs=[-Sep[k][i][j]],
                                names=["absVal_a" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_SF.linear_constraints.add(
                                lin_expr=[[["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] + ["x" + str(i + 1)]
                                           + ["x" + str(j + 1)], [1] + [mu] + [-mu]]],
                                senses="G", rhs=[Sep[k][i][j]],
                                names=["absVal_b" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        # Constraints for ranking
        for i in range(n):
            for j in range(n):
                if j > i:
                    prob_SF.linear_constraints.add(
                        lin_expr=[
                            [["y" + str(i + 1) + "," + str(j + 1)] + ["y" + str(j + 1) + "," + str(i + 1)],
                             [1] + [1]]],
                        senses="G", rhs=[1],
                        names=["dispref" + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j != k != i:
                        prob_SF.linear_constraints.add(
                            lin_expr=[[["y" + str(i + 1) + "," + str(j + 1)] + ["y" + str(k + 1) + "," + str(j + 1)]
                                       + ["y" + str(i + 1) + "," + str(k + 1)], [1] + [-1] + [-1]]],
                            senses="G", rhs=[-1],
                            names=["ttvt" + str(i + 1) + "," + str(j + 1) + "," + str(k + 1)])

        for k in range(m):
            count = 0
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    prob_SF.linear_constraints.add(
                        lin_expr=[[["h" + str(k + 1) + "," + str(i + 1)] + [("y" + str(i + 1) + "," + str(j + 1))
                                                                            for j in range(n) if i != j],
                                   [1] + [1 for j in range(n) if i != j]]],
                        senses="G", rhs=[(n - prof.evs[k].Ran[count])],
                        names=["sf1" + str(k + 1) + "," + str(i + 1)])
                    count += 1

        for k in range(m):
            count = 0
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    prob_SF.linear_constraints.add(
                        lin_expr=[[["h" + str(k + 1) + "," + str(i + 1)] + [("y" + str(i + 1) + "," + str(j + 1))
                                                                            for j in range(n) if i != j],
                                   [1] + [-1 for j in range(n) if i != j]]],
                        senses="G", rhs=[(- n + prof.evs[k].Ran[count])],
                        names=["sf2" + str(k + 1) + "," + str(i + 1)])
                    count += 1

        # Constraints for establishing relationship between rating and ranking
        # Following two equations are modified for image ranking problem
        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_SF.linear_constraints.add(
                        lin_expr=[
                            [["x" + str(i + 1)] + ["x" + str(j + 1)] + ["y" + str(i + 1) + "," + str(j + 1)],
                             [-1] + [1] + [-(M + mu)]]],
                        senses="L", rhs=[-mu],
                        names=["RatToRan_a" + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_SF.linear_constraints.add(
                        lin_expr=[
                            [["x" + str(i + 1)] + ["x" + str(j + 1)] + ["y" + str(i + 1) + "," + str(j + 1)],
                             [-1] + [1] + [-M]]],
                        senses="G", rhs=[-M],
                        names=["RatToRan_b" + str(i + 1) + "," + str(j + 1)])

        start_time = time.time()
        
        prob_SF.set_log_stream(None)
        prob_SF.set_error_stream(None)
        prob_SF.set_warning_stream(None)
        prob_SF.set_results_stream(None)
        
        prob_SF.solve()
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
                y[i][j] = round(prob_SF.solution.get_values("y" + str(i + 1) + "," + str(j + 1)))

    ranking = agg_rank_y(y)

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    x = []
    for i in range(n):
        x.append(round(mu * prob_SF.solution.get_values("x" + str(i + 1)) + L))

    # calibrate the rating values
    rating = calibrate_model(x, input_ratings)

    # Relative optimality gap
    Gap = prob_SF.solution.MIP.get_mip_relative_gap()

    # get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating, true_rating)
    L2 = scipy.spatial.distance.euclidean(rating, true_rating)/(max(true_rating)-min(true_rating))

    return ranking, dist, Gap, total_time, L1, L2


# ratings and ranking model with hamming distance and NPCK distance
def HD_ratings_and_ranking_model(file_name, num_obj, noise_level, lambda_rat, lambda_ran):

#    print("HD ratings and ranking")
    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()

    B, C, Sep, input_ratings = calculate_parameters(prof)

    n = prof.num_obj
    m = prof.num_jud
    L = prof.min_rat
    U = prof.max_rat
    mu = 1.0
    M = float((U - L) / mu)

    try:
        # Declare CPLEX object
        prob_HD = cplex.Cplex()

        # Declare if problem is maximization or minimization problem
        prob_HD.objective.set_sense(prob_HD.objective.sense.maximize)

        # y[i][j] : Decision variable
        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_HD.variables.add([0], [0], [1], types="I", names=["y" + str(i + 1) + "," + str(j + 1)])

        # h[k][i], z[k][i]: additional variable for ranking part
        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    prob_HD.variables.add([lambda_ran], [0], [1], types="I", names=["z" + str(k + 1) + "," + str(i + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    prob_HD.variables.add([0], [0], [cplex.infinity], types="C", names=["h" + str(k + 1) + "," + str(i + 1)])

        # x[i] : Decision variable
        for i in range(n):
            prob_HD.variables.add([0], [0], [math.ceil((U - L) / mu)], types="I", names=["x" + str(i + 1)])

        # t[k][i][j]: additional variable for rating part
        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_HD.variables.add([-4 * lambda_rat * C[k]], [0], [cplex.infinity], types="C",
                                                  names=["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        # Add constraints
        # Constraints for rating
        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_HD.linear_constraints.add(
                                lin_expr=[[["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] + [
                                    "x" + str(i + 1)]
                                           + ["x" + str(j + 1)], [1] + [-mu] + [mu]]],
                                senses="G", rhs=[-Sep[k][i][j]],
                                names=["absVal_a" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_HD.linear_constraints.add(
                                lin_expr=[[["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] + [
                                    "x" + str(i + 1)]
                                           + ["x" + str(j + 1)], [1] + [mu] + [-mu]]],
                                senses="G", rhs=[Sep[k][i][j]],
                                names=["absVal_b" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        # Constraints for ranking
        for i in range(n):
            for j in range(n):
                if j > i:
                    prob_HD.linear_constraints.add(
                        lin_expr=[
                            [["y" + str(i + 1) + "," + str(j + 1)] + ["y" + str(j + 1) + "," + str(i + 1)],
                             [1] + [1]]],
                        senses="G", rhs=[1],
                        names=["dispref" + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j != k != i:
                        prob_HD.linear_constraints.add(
                            lin_expr=[[["y" + str(i + 1) + "," + str(j + 1)] + ["y" + str(k + 1) + "," + str(j + 1)]
                                       + ["y" + str(i + 1) + "," + str(k + 1)], [1] + [-1] + [-1]]],
                            senses="G", rhs=[-1],
                            names=["ttvt" + str(i + 1) + "," + str(j + 1) + "," + str(k + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    prob_HD.linear_constraints.add(
                        lin_expr=[[["h" + str(k + 1) + "," + str(i + 1)] + ["z" + str(k + 1) + "," + str(i + 1)],
                                   [1] + [1000]]],
                        senses="L", rhs=[1000],
                        names=["hd1_" + str(k + 1) + "," + str(i + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    prob_HD.linear_constraints.add(
                        lin_expr=[[["h" + str(k + 1) + "," + str(i + 1)] + ["z" + str(k + 1) + "," + str(i + 1)],
                                   [1] + [1000]]],
                        senses="G", rhs=[1],
                        names=["hd2_" + str(k + 1) + "," + str(i + 1)])

        for k in range(m):
            count = 0
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    prob_HD.linear_constraints.add(
                        lin_expr=[[["h" + str(k + 1) + "," + str(i + 1)] + [("y" + str(i + 1) + "," + str(j + 1)) for j
                                                                            in range(n) if i != j],
                                   [1] + [-1 for j in range(n) if i != j]]],
                        senses="G", rhs=[-n + prof.evs[k].Ran[count]],
                        names=["hd3_" + str(k + 1) + "," + str(i + 1)])
                    count += 1

        for k in range(m):
            count = 0
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    prob_HD.linear_constraints.add(
                        lin_expr=[
                            [["h" + str(k + 1) + "," + str(i + 1)] + [("y" + str(i + 1) + "," + str(j + 1)) for j in
                                                                      range(n) if i != j],
                             [1] + [1 for j in range(n) if i != j]]],
                        senses="G", rhs=[n - prof.evs[k].Ran[count]],
                        names=["hd4_" + str(k + 1) + "," + str(i + 1)])
                    count += 1

        # Constraints for establishing relationship between rating and ranking
        # Following two equations are modified for image ranking problem
        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_HD.linear_constraints.add(
                        lin_expr=[
                            [["x" + str(i + 1)] + ["x" + str(j + 1)] + ["y" + str(i + 1) + "," + str(j + 1)],
                             [-1] + [1] + [-(M + mu)]]],
                        senses="L", rhs=[-mu],
                        names=["RatToRan_a" + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_HD.linear_constraints.add(
                        lin_expr=[
                            [["x" + str(i + 1)] + ["x" + str(j + 1)] + ["y" + str(i + 1) + "," + str(j + 1)],
                             [-1] + [1] + [-M]]],
                        senses="G", rhs=[-M],
                        names=["RatToRan_b" + str(i + 1) + "," + str(j + 1)])

        start_time = time.time()
        
        prob_HD.set_log_stream(None)
        prob_HD.set_error_stream(None)
        prob_HD.set_warning_stream(None)
        prob_HD.set_results_stream(None)
        
        prob_HD.solve()
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
                y[i][j] = round(prob_HD.solution.get_values("y" + str(i + 1) + "," + str(j + 1)))

    ranking = agg_rank_y(y)

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    x = []
    for i in range(n):
        x.append(round(mu * prob_HD.solution.get_values("x" + str(i + 1)) + L))

    # calibrate the rating values
    rating = calibrate_model(x, input_ratings)

    # Relative optimality gap
    Gap = prob_HD.solution.MIP.get_mip_relative_gap()

    # get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating, true_rating)
    L2 = scipy.spatial.distance.euclidean(rating, true_rating)/(max(true_rating)-min(true_rating))

    return ranking, dist, Gap, total_time, L1, L2


# ratings and ranking model with chebyshev's distance and NPCK distance
def CD_ratings_and_ranking_model(file_name, num_obj, noise_level, lambda_rat, lambda_ran):
#    print("CD ratings and ranking")
    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()

    B, C, Sep, input_ratings = calculate_parameters(prof)

    n = prof.num_obj
    m = prof.num_jud
    L = prof.min_rat
    U = prof.max_rat
    mu = 1.0
    M = float((U - L) / mu)

    try:
        # Declare CPLEX object
        prob_CD = cplex.Cplex()

        # Declare if problem is maximization or minimization problem
        prob_CD.objective.set_sense(prob_CD.objective.sense.minimize)

        # y[i][j] : Decision variable
        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_CD.variables.add([0], [0], [1], types="I", names=["y" + str(i + 1) + "," + str(j + 1)])

        # h[k]: additional variable for ranking part
        for k in range(m):
            prob_CD.variables.add([1], [-cplex.infinity], [cplex.infinity], types="C", names=["h" + str(k + 1)])

        # x[i] : Decision variable
        for i in range(n):
            prob_CD.variables.add([0], [0], [math.ceil((U - L) / mu)], types="I", names=["x" + str(i + 1)])

        # t[k][i][j]: additional variable for rating part
        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_CD.variables.add([-4 * lambda_rat * C[k]], [0], [cplex.infinity],
                                                  types="C",
                                                  names=["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        # Add constraints
        # Constraints for rating
        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_CD.linear_constraints.add(
                                lin_expr=[[["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] + [
                                    "x" + str(i + 1)]
                                           + ["x" + str(j + 1)], [1] + [-mu] + [mu]]],
                                senses="G", rhs=[-Sep[k][i][j]],
                                names=["absVal_a" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        for k in range(m):
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    for j in range(n):
                        if (j + 1) in prof.evs[k].Vsub and j > i:
                            prob_CD.linear_constraints.add(
                                lin_expr=[[["t" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)] + [
                                    "x" + str(i + 1)]
                                           + ["x" + str(j + 1)], [1] + [mu] + [-mu]]],
                                senses="G", rhs=[Sep[k][i][j]],
                                names=["absVal_b" + str(k + 1) + "," + str(i + 1) + "," + str(j + 1)])

        # Constraints for ranking
        for i in range(n):
            for j in range(n):
                if j > i:
                    prob_CD.linear_constraints.add(
                        lin_expr=[
                            [["y" + str(i + 1) + "," + str(j + 1)] + ["y" + str(j + 1) + "," + str(i + 1)], [1] + [1]]],
                        senses="G", rhs=[1],
                        names=["dispref" + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j != k != i:
                        prob_CD.linear_constraints.add(
                            lin_expr=[[["y" + str(i + 1) + "," + str(j + 1)] + ["y" + str(k + 1) + "," + str(j + 1)]
                                       + ["y" + str(i + 1) + "," + str(k + 1)], [1] + [-1] + [-1]]],
                            senses="G", rhs=[-1],
                            names=["ttvt" + str(i + 1) + "," + str(j + 1) + "," + str(k + 1)])

        for k in range(m):
            count = 0
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    prob_CD.linear_constraints.add(
                        lin_expr=[[["h" + str(k + 1)] + [("y" + str(i + 1) + "," + str(j + 1)) for j in range(n) if i != j],
                                   [1] + [-1 for j in range(n) if i != j]]],
                        senses="G", rhs=[- n + prof.evs[k].Ran[count]],
                        names=["l_inf_1" + str(k + 1) + "," + str(i + 1)])
                    count += 1

        for k in range(m):
            count = 0
            for i in range(n):
                if (i + 1) in prof.evs[k].Vsub:
                    prob_CD.linear_constraints.add(
                        lin_expr=[[["h" + str(k + 1)] + [("y" + str(i + 1) + "," + str(j + 1)) for j in range(n) if i != j],
                                   [1] + [1 for j in range(n) if i != j]]],
                        senses="G", rhs=[n - prof.evs[k].Ran[count]],
                        names=["l_inf_2" + str(k + 1) + "," + str(i + 1)])
                    count += 1

        # Constraints for establishing relationship between rating and ranking
        # Following two equations are modified for image ranking problem
        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_CD.linear_constraints.add(
                        lin_expr=[
                            [["x" + str(i + 1)] + ["x" + str(j + 1)] + ["y" + str(i + 1) + "," + str(j + 1)],
                             [-1] + [1] + [-(M + mu)]]],
                        senses="L", rhs=[-mu],
                        names=["RatToRan_a" + str(i + 1) + "," + str(j + 1)])

        for i in range(n):
            for j in range(n):
                if i != j:
                    prob_CD.linear_constraints.add(
                        lin_expr=[
                            [["x" + str(i + 1)] + ["x" + str(j + 1)] + ["y" + str(i + 1) + "," + str(j + 1)],
                             [-1] + [1] + [-M]]],
                        senses="G", rhs=[-M],
                        names=["RatToRan_b" + str(i + 1) + "," + str(j + 1)])

        start_time = time.time()
        
        prob_CD.set_log_stream(None)
        prob_CD.set_error_stream(None)
        prob_CD.set_warning_stream(None)
        prob_CD.set_results_stream(None)
        
        prob_CD.solve()
        
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
                y[i][j] = round(prob_CD.solution.get_values("y" + str(i + 1) + "," + str(j + 1)))

    ranking = agg_rank_y(y)

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    x = []
    for i in range(n):
        x.append(round(mu * prob_CD.solution.get_values("x" + str(i + 1)) + L))

    # calibrate the rating values
    rating = calibrate_model(x, input_ratings)

    # Relative optimality gap
    Gap = prob_CD.solution.MIP.get_mip_relative_gap()

    # get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating, true_rating)
    L2 = scipy.spatial.distance.euclidean(rating, true_rating)

    return ranking, dist, Gap, total_time, L1, L2

def ratings_only_model(file_name, num_obj, noise_level):
#    print("ratings only")    
    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()
        
    B, C, Sep, input_ratings = calculate_parameters(prof)

    n = prof.num_obj
    m = prof.num_jud
    L = prof.min_rat
    U = prof.max_rat
    R = U-L

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
                                prob_FM.variables.add([C[k]], [0], [cplex.infinity], types="C",
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
        
        prob_FM.set_log_stream(None)
        prob_FM.set_error_stream(None)
        prob_FM.set_warning_stream(None)
        prob_FM.set_results_stream(None)
        
        prob_FM.solve()
        end_time = time.time()
        total_time = end_time - start_time

    except CplexError as exc:
        print(exc)
        sys.exit()

    # Used to easily calculate the aggregate rating
    x = []
    for i in range(n):
        x.append(round(prob_FM.solution.get_values("x" + str(i + 1))))
    
    ranking = agg_rank_x(x)

    # calibration step
    rating = calibrate_model(x, input_ratings)

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    # Relative optimality gap
    Gap = prob_FM.solution.MIP.get_mip_relative_gap()


    #get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating,true_rating)

    # distance normalized by the maximum and minimum element of ground truth
    L2 = scipy.spatial.distance.euclidean(rating,true_rating)/(max(true_rating)-min(true_rating))
#    L2 = sum([i**2 for i in (rating-true_rating)/(max(true_rating)-min(true_rating))])

    # distance normalized by the maximum and minimum element of the profile
#    L2 = scipy.spatial.distance.euclidean(rating,true_rating)/R
#    L2 = sum([i**2 for i in (rating-true_rating)/R])

    return ranking, rating, dist, Gap, total_time, L1, L2

def averages(file_name, num_obj, noise_level):
#    print("averages")
    with open(file_name, "r") as read_file:
        data = json.load(read_file)
    
    rankings = []
    ratings = []
    rankingAverages = [0,0,0,0]
    ratingAverages = [0,0,0,0]
    responseCount = 0
    rat_min = 10000
    rat_max = 0

    #collect all rankings and ratings from data
    for response in data:
            if response['noiseLevel'] == noise_level:
                ranking = response['ranking']
                rating = response['rating']

                rankings.append(ranking)
                ratings.append(rating)
                groundtruth = response['groundtruth']
                responseCount +=1

    #find rating
    for rate in ratings:
        for i in range(num_obj):
            ratingAverages[i] += rate[i]
            
        if max(rate) > rat_max:
            rat_max = max(rate)
        if min(rate) < rat_min:
            rat_min = min(rate)
    for i in range(len(rankingAverages)):
        sum = ratingAverages[i]
        avg = sum / responseCount
        ratingAverages[i] = avg

    #get ranking based on ratings
    index = 1
    rankingAverages = [0,0,0,0]
    temp = list(ratingAverages)
    for i in range(num_obj):
        maxpos = temp.index(max(temp))
        rankingAverages[maxpos] = index
        temp[maxpos] = -1
        index += 1

    dist = ks_GT(rankingAverages, g_truth(groundtruth))
 

    #find rating
    for rate in ratings:
        for i in range(num_obj):
            ratingAverages[i] += rate[i]

        if max(rate) > rat_max:
            rat_max = max(rate)
        if min(rate) < rat_min:
            rat_min = min(rate)
            
    for i in range(len(rankingAverages)):
        sum = ratingAverages[i]
        avg = sum / responseCount
        ratingAverages[i] = avg

    #get ranking based on ratings
    index = 1
    rankingAverages = [0,0,0,0]
    temp = list(ratingAverages)
    for i in range(num_obj):
        maxpos = temp.index(max(temp))
        rankingAverages[maxpos] = index
        temp[maxpos] = -1
        index += 1

    dist = ks_GT(rankingAverages, g_truth(groundtruth))

    L1 = scipy.spatial.distance.cityblock(ratingAverages,groundtruth)
    # distance normalized by the maximum and minimum element of ground truth
    L2 = scipy.spatial.distance.euclidean(ratingAverages,groundtruth)/(max(groundtruth)-min(groundtruth))
    
    # distance normalized by the maximum and minimum element of the profile.
#    L2 = scipy.spatial.distance.euclidean(rating,groundtruth)/(rat_max-rat_min)
#    L2 = np.sum([i**2 for i in (np.array(ratingAverages)-np.array(groundtruth))/(rat_max-rat_min)])
    return rankingAverages, ratingAverages, dist, L1, L2

def maximin_model(file_name, num_obj, noise_level):

#    print("maximin")
    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    ranking = maximin(prof.sci.maximin)

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    return ranking, dist

def copeland_model(file_name, num_obj, noise_level):
    
#    print("copeland")
    prof = Profile(num_obj)

    prof.assign_rank(file_name, noise_level)
    prof.set_SCI()
    prof.calc_instance_stats()

    ranking = copeland(prof.sci.maximin)

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    return ranking, dist

#input_file = '../datafiles/responseData 12-06-2019.json'
input_file = 'data.json'

NOISE_LEVEL = [1,2,4,8,16,32,64,128]
lambda_rat = 1
lambda_ran = 1

OA = [0] * len(NOISE_LEVEL)
CA = [0]* len(NOISE_LEVEL)
RR = [0]* len(NOISE_LEVEL)
SD = [0]* len(NOISE_LEVEL)
CD = [0]* len(NOISE_LEVEL)
HD = [0]* len(NOISE_LEVEL)
SF = [0]* len(NOISE_LEVEL)
A = [0]* len(NOISE_LEVEL)
MM = [0]* len(NOISE_LEVEL)
CL = [0]* len(NOISE_LEVEL)

CA_L2_total = [0]* len(NOISE_LEVEL)
RR_L2_total = [0]* len(NOISE_LEVEL)
SD_L2_total = [0]* len(NOISE_LEVEL)
CD_L2_total = [0]* len(NOISE_LEVEL)
HD_L2_total = [0]* len(NOISE_LEVEL)
SF_L2_total = [0]* len(NOISE_LEVEL)
A_L2_total = [0]* len(NOISE_LEVEL)


for noise_level in NOISE_LEVEL:
    objects = 4    
    
    # Ratings and ranking model
    RR_rank, RR_rat, RR_dist, RR_gap, RR_time, RR_L1, RR_L2 = rating_and_ranking_model\
                                                            (input_file, objects, noise_level, lambda_rat, lambda_ran)
                                                            
    # Rankings only model
    OA_rank, OA_dist, OA_gap, OA_time = ranking_only_model(input_file, objects, noise_level)
    
    # Ratings only model (Fishbain moreno model)
    CA_rank, CA_rat, CA_dist, CA_gap, CA_time, CA_L1, CA_L2 = ratings_only_model(input_file, objects, noise_level)

    # Separation deviation model
    SD_rank, SD_rat, SD_dist, SD_gap, SD_time, SD_L1, SD_L2 = separation_deviation_model(input_file, objects, noise_level, lambda_rat, lambda_ran)

#    CD_rank, CD_rat, CD_dist, CD_gap, CD_time, CD_L1, CD_L2 = CD_ratings_and_ranking_model(input_file, objects, noise_level, lambda_rat, lambda_ran)

    SF_rank, SF_dist, SF_gap, SF_time, SF_L1, SF_L2 = SF_ratings_and_ranking_model(input_file, objects, noise_level, lambda_rat, lambda_ran) 
    
    HD_rank, HD_dist, HD_gap, HD_time, HD_L1, HD_L2 = HD_ratings_and_ranking_model(input_file, objects, noise_level, lambda_rat, lambda_ran)

    # Averages
    A_rank, A_rat, A_dist, A_L1, A_L2 = averages(input_file, objects, noise_level)
    
    # Maximin
    MM_rank, MM_dist = maximin_model(input_file, objects, noise_level)
    
    # Minimax
    CL_rank, CL_dist = copeland_model(input_file, objects, noise_level)
    
    OA[int(math.log(noise_level,2))] = OA_dist
    CA[int(math.log(noise_level,2))], CA_L2_total[int(math.log(noise_level,2))] = CA_dist, CA_L2
    RR[int(math.log(noise_level,2))], RR_L2_total[int(math.log(noise_level,2))] = RR_dist, RR_L2
    SD[int(math.log(noise_level,2))], SD_L2_total[int(math.log(noise_level,2))] = SD_dist, SD_L2
#    CD[int(math.log(noise_level,2))], CD_L2_total[int(math.log(noise_level,2))] = CD_dist, CD_L2    
    HD[int(math.log(noise_level,2))], HD_L2_total[int(math.log(noise_level,2))] = HD_dist, HD_L2     
    SF[int(math.log(noise_level,2))], SF_L2_total[int(math.log(noise_level,2))] = SF_dist, SF_L2   
    A[int(math.log(noise_level,2))], A_L2_total[int(math.log(noise_level,2))] = A_dist, A_L2
    MM[int(math.log(noise_level,2))] = MM_dist
    CL[int(math.log(noise_level,2))] = CL_dist

##################################################################################
#   Calculate the distance between the predicted ranking and ground truth (ranking)
##################################################################################

# X-axis: noise level, each bar represents the aggregation methods
objects = (1, 2, 4, 8, 16, 32, 64, 128)
plt.xticks(np.arange(8), objects)
plt.title('Prediction accuracy vs. Noise level (Ranking)')
plt.xlabel('Noise level')
plt.ylabel('d_KS(predicted,groundtruth)')

ax = plt.subplot(111)
ax.bar(np.arange(8)-0.4, OA, width=0.1, color='b', align='center', label='Rank')
ax.bar(np.arange(8)-0.3, CA, width=0.1, color='g', align='center', label='Rating')
ax.bar(np.arange(8)-0.2, RR, width=0.1, color='r', align='center', label='Rank+Rat')
ax.bar(np.arange(8)-0.1, SD, width=0.1, color='c', align='center', label='S-D')
ax.bar(np.arange(8), HD, width=0.1, color='violet', align='center', label='Hamming')
ax.bar(np.arange(8)+0.1, SF, width=0.1, color='saddlebrown', align='center', label='Spearman')
ax.bar(np.arange(8)+0.2, A, width=0.1, color='k', align='center', label='Avg')
ax.bar(np.arange(8)+0.3, MM, width=0.1, color='y', align='center', label='Maximin')
ax.bar(np.arange(8)+0.4, CL, width=0.1, color='m', align='center', label='Copeland')

ax.legend()

plt.grid(b=None, which='major', axis='y')
plt.show()

# X-axis: aggregation methods, each bar represents the noise level
objects = ('Rank', 'Rat', 'Rank+Rat', 'S-D', 'Hamm', 'Spear', 'Avg', 'Maximin', 'Cope')
my_array = np.array([OA, CA, RR, SD, HD, SF, A, MM, CL])
my_array = np.transpose(my_array)

plt.xticks(np.arange(9), objects)
plt.title('Prediction accuracy vs. Aggregation Methods (Ranking)')
plt.xlabel('Aggregation Methods')
plt.ylabel('d_KS(predicted,groundtruth)')

ax = plt.subplot(111)
ax.bar(np.arange(9)-0.3, my_array[0,:], width=0.1, color='0.9', align='center', label = 'noise level = 1')
ax.bar(np.arange(9)-0.2, my_array[1,:], width=0.1, color= '0.8', align='center', label = 'noise level = 2')
ax.bar(np.arange(9)-0.1, my_array[2,:], width=0.1, color='0.7', align='center', label = 'noise level = 4')
ax.bar(np.arange(9), my_array[3,:], width=0.1, color='0.6', align='center', label = 'noise level = 8')
ax.bar(np.arange(9)+0.1, my_array[4,:], width=0.1, color='0.4', align='center', label = 'noise level = 16')
ax.bar(np.arange(9)+0.2, my_array[5,:], width=0.1, color='0.2', align='center', label = 'noise level = 32')
ax.bar(np.arange(9)+0.3, my_array[6,:], width=0.1, color='0.1', align='center', label = 'noise level = 64')
ax.bar(np.arange(9)+0.4, my_array[7,:], width=0.1, color='0', align='center', label = 'noise level = 128')
ax.legend(prop={'size': 8})


ax.legend(prop={'size': 9})

plt.show()


##################################################################################
#   Calculate the distance between the predicted rating and ground truth (rating)
##################################################################################

objects = (1, 2, 4, 8, 16, 32, 64, 128)
plt.xticks(np.arange(8), objects)
plt.title('Prediction accuracy vs. Noise level (Rating)')
plt.xlabel('Noise level')
plt.ylabel('NormalizedEuclideanDist(predicted,groundtruth)')

ax = plt.subplot(111)
ax.bar(np.arange(8)-0.3, CA_L2_total, width=0.1, color='g', align='center', label='Rating')
ax.bar(np.arange(8)-0.2, RR_L2_total, width=0.1, color='r', align='center', label='Rank+Rat')
ax.bar(np.arange(8)-0.1, HD_L2_total, width=0.1, color='violet', align='center', label='Hamming')
ax.bar(np.arange(8), SF_L2_total, width=0.1, color='saddlebrown', align='center', label='Spearman')
ax.bar(np.arange(8)+0.1, SD_L2_total, width=0.1, color='c', align='center', label='S-D')
ax.bar(np.arange(8)+0.2, A_L2_total, width=0.1, color='k', align='center', label='Avg')
ax.legend()

plt.grid(b=None, which='major', axis='y')
plt.show()


objects = ('Rating', 'Rank+Rat', 'S-D', 'Hamming', 'Spearman', 'Avg')
my_array = np.array([CA_L2_total, RR_L2_total, SD_L2_total, HD_L2_total, SF_L2_total, A_L2_total])
my_array = np.transpose(my_array)

plt.xticks(np.arange(6), objects)
plt.title('Prediction accuracy vs. Aggregation methods (Rating)')
plt.xlabel('Aggregation Methods')
plt.ylabel('NormalizedEuclideanDist(predicted,groundtruth)')

ax = plt.subplot(111)
ax.bar(np.arange(6)-0.3, my_array[0,:], width=0.1, color='0.9', align='center', label = 'noise level = 1')
ax.bar(np.arange(6)-0.2, my_array[1,:], width=0.1, color= '0.8', align='center', label = 'noise level = 2')
ax.bar(np.arange(6)-0.1, my_array[2,:], width=0.1, color='0.7', align='center', label = 'noise level = 4')
ax.bar(np.arange(6), my_array[3,:], width=0.1, color='0.6', align='center', label = 'noise level = 8')
ax.bar(np.arange(6)+0.1, my_array[4,:], width=0.1, color='0.4', align='center', label = 'noise level = 16')
ax.bar(np.arange(6)+0.2, my_array[5,:], width=0.1, color='0.2', align='center', label = 'noise level = 32')
ax.bar(np.arange(6)+0.3, my_array[6,:], width=0.1, color='0.1', align='center', label = 'noise level = 64')
ax.bar(np.arange(6)+0.4, my_array[7,:], width=0.1, color='0.0', align='center', label = 'noise level = 128')
ax.legend(prop={'size': 8})

plt.show()
