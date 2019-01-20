from random import randint
import random
import os
import math

# Functions returns array of clauses [[1,2,3], [-1,2], ... ]
# and number of literals(variables)
def extract_clauses_literals_from_file(filename):
    with open(filename, "r") as input:
        header = input.readline().split(" ")
        literals = int(header[2].rstrip())

        text = input.readlines()

        for i in range(len(text)):
            text[i] = text[i].split(" ")[:-1]
            text[i] = [int(x) for x in text[i]]

        return (text, literals)

# Returns valuation list for num = 0 and k = 3 : 000
#                            num = 1 and k = 3 : 001
#                            ...
#                            num = 7 and k = 3 : 111
def binary_list(i, num_literals):
    bin_str = bin(i)[2:].zfill(num_literals)
    bin_list = [int(x) for x in bin_str]

    return bin_list

# Returns 1 if clause is true (satistied) or 0 if not satisfied.
def satisfied_clause(valuation_list, clause):
    len_clause = len(clause)
    values = [1-valuation_list[-clause[i]-1] if (clause[i]<0) else valuation_list[clause[i]-1]
                for i in range(0, len_clause)]

    return 0 if sum(values)==0 else 1

# For i combination returns valuation list and number of true clauses
def solution(i, literals, clauses):
    valuation_list = binary_list(i, literals)
    num_true_clauses = 0

    for c in clauses:
        num_true_clauses += satisfied_clause(valuation_list, c)

    return valuation_list, num_true_clauses

def solution_based_on_val_list(val_list, clauses):
    num_true_clauses = 0
    for c in clauses:
        num_true_clauses += satisfied_clause(val_list, c)
    return num_true_clauses

# BRUTE FORCE ALGORITHM
# For each combination from 0 to 2^num_of_vars count
# number of true clauses
def brute_force_algorithm(clauses, literals):
    n = 2**literals
    num_of_clauses = len(clauses)

    max = 0
    res_val_list = []

    for i in range(n):
        valuation_list, curr_max = solution(i, literals, clauses)

        if curr_max > max:
            max = curr_max
            res_val_list = valuation_list

        if max == num_of_clauses:
            break

    return (max, res_val_list)

# RANDOM ALGORITHM
# For given number of iterations gives random number
# for which we check number of true clauses.
# Does not garantee best solution!
def random_algorithm(clauses, literals, num_of_iters):
    n = 2**literals
    num_of_clauses = len(clauses)

    max = 0
    res_val_list = []

    for i in range(num_of_iters):
        random_num = randint(0,n-1)
        valuation_list, curr_max = solution(random_num, literals, clauses)

        if curr_max > max:
            max = curr_max
            res_val_list = valuation_list

        if max == num_of_clauses:
            break

    return (max, res_val_list)


# SIMULATED ANNEALING ALGORITHM (Simulirano kaljenje)
#
#

# TODO: MAYBE WE CAN DO THIS DIFFERENTLY - THINK
# Initialize random solution
# Make initial valuation list with random picked combination (i)
# Our init max is number of true clauses in picked combination
def initialize_for_SA(clauses, literals):
    n = 2**literals
    random_i = randint(0, n-1)

    val_list = binary_list(random_i, literals)
    num_true_clauses = 0

    for c in clauses:
        num_true_clauses += satisfied_clause(val_list, c)

    return num_true_clauses, val_list

# TODO: MOST IMPORTANT FUNC OF ALGORITHM - HOW TO FIND NEIGHBOUR - FIND BETTER WAY
# For now it pickes random position in valuation list and swaps values on pos and pos-1
# Returns new valuation list and number of true clauses
def invert_val_list_based_on_pos(curr_val_list, pos):
    if pos == 0 or pos == (len(curr_val_list)-1):
        curr_val_list[0], curr_val_list[len(curr_val_list)-1] = curr_val_list[len(curr_val_list)-1], curr_val_list[0]
    else:
        curr_val_list[pos], curr_val_list[pos-1] = curr_val_list[pos-1], curr_val_list[pos]

    return curr_val_list

def invert_val_list_for_SA(curr_val_list):
    pos = randint(0, len(curr_val_list)-1)
    curr_val_list = invert_val_list_based_on_pos(curr_val_list, pos)
    return curr_val_list, pos

def restore_invert_for_SA(curr_val_list, pos):
    curr_val_list = invert_val_list_based_on_pos(curr_val_list, pos)
    return curr_val_list

def simulated_annealing_algorithm(clauses, literals, num_of_iters):
    init_max, init_val_list = initialize_for_SA(clauses, literals)

    curr_max, curr_val_list = init_max, init_val_list

    max = curr_max
    res_val_list = []

    num_of_clauses = len(clauses)

    for i in range(num_of_iters):
        new_val_list, pos = invert_val_list_for_SA(curr_val_list)
        new_max = solution_based_on_val_list(new_val_list, clauses)

        if new_max > curr_max:
            curr_max = new_max
            curr_val_list = new_val_list

        elif new_max <= curr_max:
            p = 1.0/math.pow(i+1, 0.5)
            q = random.uniform(0,1)

            # TODO check sign!!!
            if p > q:
                curr_max = new_max
                curr_val_list = new_val_list
            else:
                curr_val_list = restore_invert_for_SA(curr_val_list, pos)

        if new_max > max:
            max = new_max
            res_val_list = new_val_list

        if max == num_of_clauses:
            break

    return (max, res_val_list)

def run_brute_force(filename):
    clauses, literals = extract_clauses_literals_from_file(os.path.abspath(filename))
    return brute_force_algorithm(clauses, literals)

def run_random(filename, num_of_iters):
    clauses, literals = extract_clauses_literals_from_file(os.path.abspath(filename))
    return random_algorithm(clauses, literals, num_of_iters)

def run_simulated_annealing(filename, num_of_iters):
    clauses, literals = extract_clauses_literals_from_file(os.path.abspath(filename))
    return simulated_annealing_algorithm(clauses, literals, num_of_iters)

def main():
    max, val_list = run_brute_force("examples/input_easy.cnf")
    print(max, val_list)

    max, val_list = run_random("examples/input_sudoku.cnf", 400)
    print(max, val_list)

    max, val_list = run_simulated_annealing("examples/input_sudoku.cnf", 400)
    print(max, val_list)

if __name__ == "__main__":
    main()
