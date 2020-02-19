from __future__ import division
import json
import math
import sys
import numpy as np
import cplex
from cplex.exceptions import CplexError
import time
import scipy
from scipy.spatial import distance
from operator import truediv
from itertools import combinations 
import scipy.stats as ss


def val2rank(n):
    return n - 49

class Sampler:

    def __init__(self, n, num_batches, input_file, obj, frames):
        self.sampleSize = n
        self.array = [i for i in range(num_batches)]
        self.batches = [0] * num_batches
        self.input_file = input_file
        self.objects = obj
        self.frames = frames

    def sample_rating_and_ranking_model(self):

        combinations_array = list(combinations(self.array, self.sampleSize))

        RR_dist_sum = 0
        RR_L2_sum = 0
    
        combinationCount = 0

        for comb in combinations_array:

            for q in range(len(self.batches)):
                if (q in comb):
                    self.batches[q] = 1
                else:
                    self.batches[q] = 0

            # Ratings and ranking model
            RR_rank, RR_rat, RR_dist, RR_gap, RR_time, RR_L1, RR_L2 = rating_and_ranking_model(self.input_file, self.objects, self.batches, self.frames, lambda_rat, lambda_ran)
            RR_dist_sum += RR_dist
            RR_L2_sum += RR_L2

            combinationCount +=1
        
        RR_dist = RR_dist_sum/combinationCount
        RR_L2 = RR_L2_sum/combinationCount

        return RR_dist, RR_L2, RR_gap

    def sample_ranking_only_model(self):

        combinations_array = list(combinations(self.array, self.sampleSize))

        OA_dist_sum = 0

        combinationCount = 0

        for comb in combinations_array:

            for q in range(len(self.batches)):
                if (q in comb):
                    self.batches[q] = 1
                else:
                    self.batches[q] = 0

            # Rankings only model
            OA_rank, OA_dist, OA_gap, OA_time = ranking_only_model(self.input_file, self.objects, self.batches, self.frames)
            OA_dist_sum += OA_dist

            combinationCount +=1

        OA_dist = OA_dist_sum/combinationCount

        return OA_dist

    def sample_ratings_only_model(self):

        combinations_array = list(combinations(self.array, self.sampleSize))

        CA_dist_sum = 0
        CA_L2_sum = 0

        combinationCount = 0

        for comb in combinations_array:

            for q in range(len(self.batches)):
                if (q in comb):
                    self.batches[q] = 1
                else:
                    self.batches[q] = 0

            # Ratings only model (Fishbain moreno model)
            CA_rank, CA_rat, CA_dist, CA_gap, CA_time, CA_L1, CA_L2 = ratings_only_model(self.input_file, self.objects, self.batches, self.frames)
            CA_dist_sum += CA_dist
            CA_L2_sum += CA_L2

            combinationCount +=1

        CA_dist = CA_dist_sum/combinationCount
        CA_L2 = CA_L2_sum/combinationCount

        return CA_dist, CA_L2

    def sample_separation_deviation_model(self):

        combinations_array = list(combinations(self.array, self.sampleSize))

        SD_dist_sum = 0
        SD_L2_sum = 0

        combinationCount = 0

        for comb in combinations_array:

            for q in range(len(self.batches)):
                if (q in comb):
                    self.batches[q] = 1
                else:
                    self.batches[q] = 0

            # Separation deviation model
            SD_rank, SD_rat, SD_dist, SD_gap, SD_time, SD_L1, SD_L2 = separation_deviation_model(self.input_file, self.objects, self.batches, self.frames, 1, 1)
            SD_dist_sum += SD_dist
            SD_L2_sum += SD_L2

            combinationCount +=1

        SD_dist = SD_dist_sum/combinationCount
        SD_L2 = SD_L2_sum/combinationCount

        return SD_dist, SD_L2

    def sample_averages(self):

        combintations_array = list(combinations(self.array, self.sampleSize))

        A_dist_sum = 0
        A_L2_sum = 0

        combinationCount = 0

        for comb in combintations_array:

            for q in range(len(self.batches)):
                if (q in comb):
                    self.batches[q] = 1
                else:
                    self.batches[q] = 0

            #averages
            A_rank, A_rat, A_dist, A_L1, A_L2 = averages(self.input_file, self.objects, self.batches, self.frames)
            A_dist_sum += A_dist
            A_L2_sum += A_L2

            combinationCount +=1
        
        A_dist = A_dist_sum/combinationCount
        A_L2 = A_L2_sum/combinationCount

        return A_dist, A_L2


    def sample_SF_ratings_and_ranking_model(self):

        combinations_array = list(combinations(self.array, self.sampleSize))

        SF_dist_sum = 0
        SF_L2_sum = 0
    
        combinationCount = 0

        for comb in combinations_array:

            for q in range(len(self.batches)):
                if (q in comb):
                    self.batches[q] = 1
                else:
                    self.batches[q] = 0

            # Ratings and ranking model
            SF_rank, SF_rat, SF_dist, SF_gap, SF_time, SF_L1, SF_L2 = SF_ratings_and_ranking_model(self.input_file, self.objects, self.batches, self.frames, lambda_rat, lambda_ran)
            SF_dist_sum += SF_dist
            SF_L2_sum += SF_L2

            combinationCount +=1
        
        SF_dist = SF_dist_sum/combinationCount
        SF_L2 = SF_L2_sum/combinationCount

        return SF_dist, SF_L2, SF_gap

    def sample_HD_ratings_and_ranking_model(self):

        combinations_array = list(combinations(self.array, self.sampleSize))

        HD_dist_sum = 0
        HD_L2_sum = 0
    
        combinationCount = 0

        for comb in combinations_array:

            for q in range(len(self.batches)):
                if (q in comb):
                    self.batches[q] = 1
                else:
                    self.batches[q] = 0

            # Ratings and ranking model
            HD_rank, HD_rat, HD_dist, HD_gap, HD_time, HD_L1, HD_L2 = HD_ratings_and_ranking_model(self.input_file, self.objects, self.batches, self.frames, lambda_rat, lambda_ran)
            HD_dist_sum += HD_dist
            HD_L2_sum += HD_L2

            combinationCount +=1
        
        HD_dist = HD_dist_sum/combinationCount
        HD_L2 = HD_L2_sum/combinationCount

        return HD_dist, HD_L2, HD_gap

    def sample_maximin_model(self):

        combinations_array = list(combinations(self.array, self.sampleSize))

        MM_dist_sum = 0

        combinationCount = 0

        for comb in combinations_array:

            for q in range(len(self.batches)):
                if (q in comb):
                    self.batches[q] = 1
                else:
                    self.batches[q] = 0

            # Rankings only model
            MM_rank, MM_dist = maximin_model(self.input_file, self.objects, self.batches, self.frames)
            MM_dist_sum += MM_dist

            combinationCount +=1

        MM_dist = MM_dist_sum/combinationCount

        return MM_dist
  
    def sample_copeland_model(self):

        combinations_array = list(combinations(self.array, self.sampleSize))

        CL_dist_sum = 0

        combinationCount = 0

        for comb in combinations_array:

            for q in range(len(self.batches)):
                if (q in comb):
                    self.batches[q] = 1
                else:
                    self.batches[q] = 0

            # Rankings only model
            CL_rank, CL_dist = copeland_model(self.input_file, self.objects, self.batches, self.frames)
            CL_dist_sum += CL_dist

            combinationCount +=1

        CL_dist = CL_dist_sum/combinationCount

        return CL_dist      
    

class Evaluation:

    def __init__(self):
        self.size = 0               # Size of individual rating vector
        self.Vsub = []
        self.groundtruth = []
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
        self.min_rat = 9999
        self.max_rat = 0


    # Read datafile and assign ranking, rating and object values to each judges in evaluation
    def assign_rank(self, input_file, batches, frames):

        with open(input_file) as json_file:
            data = json.load(json_file)

        for item in data:

            #check for batch and frame

            batch = int(item['batch'])
            frame = int(item['frames'])
            if frames == frame:
                if batches[batch] == 1:

                    tempEval = Evaluation()
                    tempEval.Ran = item['rankings']
                    tempEval.Rat = item['ratings']
                    tempEval.size = len(tempEval.Ran)

                    tempEval.Vsub = [val2rank(i) for i in item['groundtruth']]
                    self.groundtruth = [50 + i for i in range(30)]

                    min_rat = np.min(tempEval.Rat)
                    max_rat = np.max(tempEval.Rat)

                    if min_rat < self.min_rat:
                        self.min_rat = min_rat

                    if max_rat > self.max_rat:
                        self.max_rat = max_rat
                    #self.max_rat = max(self.groundtruth)
                    #self.min_rat = min(self.groundtruth)

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
#            print(i, self.evs[i].Ran, self.evs[i].Rat)           
            
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
        minEntry = 9999
        count = 0
        for j in range(len(vec)):
            if r[j] == 0 and vec[j] <= minEntry:
                minEntry = vec[j]

        for j in range(len(vec)):
            if vec[j] == minEntry:
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

    for i in range(len(vec)):
        r[i] = i+1
    return r

# Calculate all the parameters
def calculate_parameters(p):
    profile = p
    num_obj = profile.num_obj
    num_jud = profile.num_jud
    B = np.zeros((num_obj, num_obj))
    L = profile.min_rat
    U = profile.max_rat
    R = U - L
    n_indiv = [0] * num_jud
    C = [0] * num_jud
    Sep = np.zeros((num_jud, num_obj, num_obj))

    for i in range(num_obj):
        for j in range(num_obj):
            B[i][j] = round(profile.sci.mat[i][j], 5)

    for j in range(num_jud):
        n_indiv[j] = float(profile.evs[j].size)
        C[j] = 1 / (4 * R * math.ceil(n_indiv[j] / 2) * math.floor(n_indiv[j] / 2))

    for k in range(num_jud):
        for i in range(profile.evs[k].size):
            for j in range(profile.evs[k].size):
                if j > i:
                    Sep[k][profile.evs[k].Vsub[i] - 1][profile.evs[k].Vsub[j] - 1] = profile.evs[k].seps[i][j]

    # input rating matrix
    rat_matrix = np.zeros((num_jud, num_obj))
    for k in range(num_jud):
        for i in range(len(profile.evs[k].Vsub)):
            rat_matrix[k][profile.evs[k].Vsub[i] - 1] = profile.evs[k].Rat[i]


    return B, C, Sep, rat_matrix


 

def callibrate_model(solns, data):

    # this is the solution vector (aggregate rating vector)
    solns = np.array(solns)

    m = cplex.Cplex()
    m.objective.set_sense(m.objective.sense.minimize)

    # number of judges and number of objects
    num_judges, num_objects = np.shape(data)

    # add the variable "c" to the model
    m.variables.add([0], [-cplex.infinity], [cplex.infinity], types="C", names=["cal"])

    for i in range(num_judges):
        for j in range(num_objects):
            m.variables.add([0], [-cplex.infinity], [cplex.infinity], types="C", names=["h_abs" + str(i) + "," + str(j)])

    # objective function has absolute value so linearize this using the following two equations
    # objective function : min |(x_i + c) - a_ij|

    for i in range(num_judges):
        for j in range(num_objects):
            if data[i, j] != 0:
                m.linear_constraints.add(
                    lin_expr=[[["h_abs" + str(i) + "," + str(j)] + ["cal"], [1] + [-1]]],
                    senses="G", rhs=[solns[j] - data[i, j]],
                    names=["abs_1" + str(i) + "," + str(j)])

    for i in range(num_judges):
        for j in range(num_objects):
            if data[i, j] != 0:
                m.linear_constraints.add(
                    lin_expr=[[["h_abs" + str(i) + "," + str(j)] + ["cal"], [1] + [1]]],
                    senses="L", rhs=[- solns[j] + data[i, j]],
                    names=["abs_2" + str(i) + "," + str(j)])


    m.set_log_stream(None)
    m.set_error_stream(None)
    m.set_warning_stream(None)
    m.set_results_stream(None)
    
    m.solve()

    cal = m.solution.get_values("cal")

    for k in range(len(solns)):
        solns[k] = solns[k] + cal

    return solns

# ratings and ranking model with NPKS and NPCK distance
def rating_and_ranking_model(file_name, num_obj, batches, frames, lambda_rat, lambda_ran):

    prof = Profile(num_obj)

    prof.assign_rank(file_name, batches, frames)
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
    #rating = cal_model(x, input_ratings)
    rating = x

    # Relative optimality gap
    Gap = prob_RR.solution.MIP.get_mip_relative_gap()

    #get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating, true_rating)
    L2 = scipy.spatial.distance.euclidean(rating, true_rating)

    return ranking, rating, dist, Gap, total_time, L1, L2


def ranking_only_model(file_name, num_obj, batches, frames):

    prof = Profile(num_obj)

    prof.assign_rank(file_name, batches, frames)
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


def separation_deviation_model(file_name, num_obj, batches, frames, lambda1, lambda2):

    prof = Profile(num_obj)

    prof.assign_rank(file_name, batches, frames)
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

    rating = x

    ground_truth = g_truth(prof.groundtruth)
    
    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    # Relative optimality gap
    Gap = prob_SD.solution.MIP.get_mip_relative_gap()

    #get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating,true_rating)
    L2 = scipy.spatial.distance.euclidean(rating,true_rating)

    return ranking, rating, dist, Gap, total_time, L1, L2



def ratings_only_model(file_name, num_obj, batches, frames):

    prof = Profile(num_obj)

    prof.assign_rank(file_name, batches, frames)
    prof.set_SCI()
    prof.calc_instance_stats()

    for i in range(prof.num_jud):
        prof.evs[i].calc_ratings_seps()

    B, C, Sep, input_ratings = calculate_parameters(prof)

    n = prof.num_obj
    m = prof.num_jud

    L = prof.min_rat
    U = prof.max_rat
    
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

    # Used to easily calculate the aggregate ranking
    x = []
    for i in range(n):
        x.append(round(prob_FM.solution.get_values("x" + str(i + 1))))

    ranking = agg_rank_x(x)


    # calibration step
    #rating = callibrate_model(x, input_ratings)
    rating = x

    ground_truth = g_truth(prof.groundtruth)

    # KS distance between the aggregate ranking and the ground truth
    dist = ks_GT(ranking, ground_truth)

    # Relative optimality gap
    Gap = prob_FM.solution.MIP.get_mip_relative_gap()

    #get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating, true_rating)
    L2 = scipy.spatial.distance.euclidean(rating, true_rating)

    return ranking, rating, dist, Gap, total_time, L1, L2


def averages(file_name, num_obj, batches, frames):

    #initialize arrays 
    gtruth = [50 + i for i in range(32)]
    rankgtruth = [32 - i for i in range(32)]

    #load in data
    with open(file_name, "r") as read_file:
        data = json.load(read_file)

    ratingSums = [0] * 32
    numberCounts = [0] * 32


    #NOTE: NEED TO FIX

    #find rating averages
    for item in data:
        for batch in range(len(batches)):
            if batched[batch] == 1:
                rating = item['ratings']
                truth = item['groundtruth'][question]

                for l in range(len(truth)):
                    idx = gtruth.index(truth[l])
                    ratingSums[idx] += rating[l]
                    numberCounts[idx] += 1

    ratingAverages = list(map(truediv, ratingSums, numberCounts))

    #get ranking based on rating averages
    index = 32
    rankingAverages = [0] * num_obj
    temp = list(ratingAverages)
    for i in range(num_obj):
        minpos = temp.index(min(temp))
        rankingAverages[minpos] = index
        temp[minpos] = 9999
        index -= 1

    dist = ks_GT(rankingAverages, rankgtruth)

    #get distance
    L1 = scipy.spatial.distance.cityblock(ratingAverages,gtruth)
    L2 = scipy.spatial.distance.euclidean(ratingAverages,gtruth)

    return rankingAverages, ratingAverages, dist, L1, L2

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
        x.append(round(mu*prob_SF.solution.get_values("x" + str(i + 1)) + L))

    # calibrate the rating values
    #rating = cal_model(x, input_ratings)
    rating = x

    # Relative optimality gap
    Gap = prob_SF.solution.MIP.get_mip_relative_gap()


    # get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating, true_rating)
    L2 = scipy.spatial.distance.euclidean(rating, true_rating)/(max(true_rating)-min(true_rating))

    return ranking, rating, dist, Gap, total_time, L1, L2


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
        x.append(round(mu*prob_HD.solution.get_values("x" + str(i + 1)) + L))

    # calibrate the rating values
    #rating = cal_model(x, input_ratings)
    rating = x

    # Relative optimality gap
    Gap = prob_HD.solution.MIP.get_mip_relative_gap()


    # get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating, true_rating)
    L2 = scipy.spatial.distance.euclidean(rating, true_rating)/(max(true_rating)-min(true_rating))

    return ranking, rating, dist, Gap, total_time, L1, L2


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
        x.append(round(mu*prob_CD.solution.get_values("x" + str(i + 1)) + L))

    # calibrate the rating values
    #rating = cal_model(x, input_ratings)
    rating = x

    # Relative optimality gap
    Gap = prob_CD.solution.MIP.get_mip_relative_gap()


    # get distances
    true_rating = prof.groundtruth
    L1 = scipy.spatial.distance.cityblock(rating, true_rating)
    L2 = scipy.spatial.distance.euclidean(rating, true_rating)

    return ranking, rating, dist, Gap, total_time, L1, L2

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


#input_file = 'responseData Incomplete 12-3.json'
input_file = '02-17-2020ratingsrankingsA1.json'
objects = 30
num_batches = 10
size = 4 #number of batches to sample
frames = 5

lambda_rat = 1 
lambda_ran = 1

start_time = time.time()

sample = Sampler(size, num_batches, input_file, objects, frames)

#print incompleteness level
print("Incompleteness level: " + str(size))

#rankings only model
OA_dist = sample.sample_ranking_only_model()
print('rankings only :')
print(OA_dist)

#ratings only model
CA_dist, CA_L2 = sample.sample_ratings_only_model()

print('ratings only:')
print(CA_dist)
print(CA_L2)

#separation deviation model
SD_dist, SD_L2 = sample.sample_separation_deviation_model()
print('separation deviation: ')
print(SD_dist)
print(SD_L2)


"""

#ratings and rankings model
RR_dist, RR_L2, RR_gap = sample.sample_rating_and_ranking_model()
print('ratings and rankings:')
print(RR_dist)
print(RR_L2)

#averages
A_dist, A_L2 = sample.sample_averages()
print('averages: ')
print(A_dist)
print(A_L2)


SF_dist, SF_L2 = sample.sample_SF_ratings_and_ranking_model() 
print('spearman footrule model')
print(SF_dist)
print(SF_L2)

HD_dist, HD_L2 = sample.sample_HD_ratings_and_ranking_model()
print('hamming distance model')
print(HD_dist)
print(HD_L2)

MM_dist = sample.sample_maximin_model()
print('maximin model: ')
print(MM_dist)

CL_dist = sample.sample_copeland_model()
print('copeland model: ')
print(CL_dist)

total_time = time.time() - start_time
print("time:", total_time)

# Ratings and ranking model
RR_rank, RR_rat, RR_dist, RR_gap, RR_time, RR_L1, RR_L2 = rating_and_ranking_model(input_file, objects, questions)
# Rankings only model
OA_rank, OA_dist, OA_gap, OA_time = ranking_only_model(input_file, objects, questions)
# Ratings only model (Fishbain moreno model)
CA_rank, CA_rat, CA_dist, CA_gap, CA_time, CA_L1, CA_L2 = ratings_only_model(input_file, objects, questions)
# Separation deviation model
SD_rank, SD_rat, SD_dist, SD_gap, SD_time, SD_L1, SD_L2 = separation_deviation_model(input_file, objects, questions, 1, 1)
#averages
A_rank, A_rat, A_dist, A_L1, A_L2 = averages(input_file, objects, questions)

print('ranking only model')
print('rank: ' + str(OA_rank))
print('distance: ' + str(OA_dist))

print('ratings and rankings model')
print('rank: ' + str(RR_rank))
print('rating: ' + str(RR_rat))
print('KS Distance: ' + str(RR_dist))
print('L2 Distance: ' + str(RR_L2))

print('rating only model')
print('rank: ' + str(CA_rank))
print('rating: ' + str(CA_rat))
print('KS Distance: ' + str(CA_dist))
print('L2 Distance: ' + str(CA_L2))

print('separation deviation model')
print('rank: ' + str(SD_rank))
print('rating: ' + str(SD_rat))
print('KS Distance: ' + str(SD_dist))
print('L2 Distance: ' + str(SD_L2))

print('averages')
print('rank: ' + str(A_rank))
print('rating: ' + str(A_rat))
print('KS Distance: ' + str(A_dist))
print('L2 Distance: ' + str(A_L2))

"""
