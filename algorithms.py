#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randint, uniform, random
import os
import math

# Functions returns array of clauses [[1,2,3], [-1,2], ... ]
# and number of literals(variables)
def clauses_from_file(filename):
    with open(filename, "r") as fin:
        #remove comments from beginning
        line = fin.readline()
        while(line.lstrip()[0] == 'c'):
            line = fin.readline()

        header = line.split(" ")
        num_literals = int(header[2].rstrip())

        lines = fin.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i].split(" ")[:-1]
            lines[i] = [int(x) for x in lines[i]]

        return (lines, num_literals)


"""
Returns binary representation of decimal number with specified width
example: dec_number = 2 and width = 3, returns: 010
"""
def binary_list(dec_number, width):
    bin_str = bin(dec_number)[2:].zfill(width)
    return [int(x) for x in bin_str]


"""
Returns True if clause is true (satisfied) or False if not satisfied.
"""
def is_clause_satisfied(valuation_list, clause):
    for literal in clause:
        if literal < 0:
            v = 1 - valuation_list[-literal - 1]
        else:
            v = valuation_list[literal - 1]

        if v == 1:
            return True

    return False


# For i combination returns valuation list and number of true clauses
def solution(i, literals, clauses):
    valuation_list = binary_list(i, literals)
    num_true_clauses = 0

    for c in clauses:
        num_true_clauses += is_clause_satisfied(valuation_list, c)

    return valuation_list, num_true_clauses

def fitness(val_list, clauses):
    num_true_clauses = 0
    for c in clauses:
        num_true_clauses += is_clause_satisfied(val_list, c)
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
        num_true_clauses += is_clause_satisfied(val_list, c)

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
        new_max = fitness(new_val_list, clauses)

        if new_max > curr_max:
            curr_max = new_max
            curr_val_list = new_val_list

        elif new_max <= curr_max:
            p = 1.0/math.pow(i+1, 0.5)
            q = uniform(0,1)

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



# PSO ALGORITHM

"""
Returns array of clauses with weights = 1
[{'clause':[1,2,3], 'w':1}, {'clause':[-1,2], 'w':1}... ]
and number of literals
"""
def w_clauses_from_file(filename):
    clauses = []
    with open(filename, "r") as fin:
        #remove comments from beginning
        line = fin.readline()
        while(line.lstrip()[0] == 'c'):
            line = fin.readline()

        header = line.split(" ")
        num_literals = int(header[2].rstrip())

        lines = fin.readlines()
        for line in lines:
            line = line.split(" ")[:-1]
            line = [int(x) for x in line]
            clauses.append({'clause':line, 'w':1})

        return (clauses, num_literals)

"""
Update clauses weight to identify the hard clauses
"""
def update_clauses_weight(clauses, best_particle):
    for clause in clauses:
        clause['w'] = clause['w'] + 1 - is_clause_satisfied(best_particle, clause['clause'])


def sigmoid(velocity):
    return 1.0/(1+math.exp(-velocity))


def fitness_PSO(val_list, clauses_with_w):
    return sum(map(lambda i: i['w'] * is_clause_satisfied(val_list, i['clause']), clauses_with_w))


"""
Initialize the population, positions and velocities
"""
def init_PSO(literals, num_particles, clauses):
    print(clauses)

    n = 2**literals

    swarm = []
    for i in range(num_particles):
        position = binary_list(randint(0, n-1), literals)
        #initial velocities from -1 to 1
        velocity = [2*random()-1 for x in range(literals)]

        particle_map = {}
        particle_map['position'] = position
        particle_map['velocity'] = velocity
        particle_map['best'] = particle_map['position']
        swarm.append(particle_map)

    #TODO remove print
    print("Init:")
    for particle in swarm:
        print(particle)
    print(10*"-")

    return swarm


def satisfied_clauses(val_list, formula):
    num_true_clauses = 0
    for c in formula:
        num_true_clauses += is_clause_satisfied(val_list, c['clause'])
    return num_true_clauses


def stop_condition(global_best, clauses, num_literals, iteration, max_iteration):
    #TODO if fitness == num_literas
    if satisfied_clauses(global_best, clauses) == num_literals or iteration > max_iteration:
        return True
    return False


def PSO(clauses, literals, num_particles, max_iteration, w = 1, c1 = 2, c2 = 2):
    swarm = init_PSO(literals, num_particles, clauses)

    best_particle_fitnes = 0
    iteration = 0
    global_best = swarm[0]['position']

    while(not stop_condition(global_best, clauses, literals, iteration, max_iteration)):

        for particle in swarm:
            #Calculate fitness
            particle_fitness = fitness_PSO(particle['position'], clauses)

            #Save the individuals highest fitness (Pg)
            #Update global best
            if particle_fitness > best_particle_fitnes:
                best_particle_fitnes = particle_fitness
                global_best = particle['position']

        #print("Best fitness:")
        #print(best_particle_fitnes)
        #print(global_best)

        #Modify velocities
        for particle in swarm:
            r1 = random();
            r2 = random();
            new_velocity = []
            for i in range(literals):
                velocity_i = w*particle['velocity'][i] + c1*r1*(particle['best'][i] - particle['position'][i]) + c2*r2*(global_best[i] - particle['position'][i])
                new_velocity.append(sigmoid(velocity_i))
            particle['velocity'] = new_velocity


        #Update the particles position
        #TODO using flight
            r = random();
            new_position = []
            for i in range(literals):
                position_i = particle['position'][i] + particle['velocity'][i]
                position_i = 1 if r < particle['velocity'][i] else 0
                new_position.append(position_i)
            particle['position'] = new_position

            #Update particle best
            if fitness_PSO(particle['position'], clauses) > fitness_PSO(particle['best'], clauses):
                particle['best'] = particle['position']

            curr_fitness = fitness_PSO(particle['best'], clauses)
            if curr_fitness > best_particle_fitnes:
                best_particle_fitnes = curr_fitness
                global_best = particle['best']

            #print(particle)


        update_clauses_weight(clauses, global_best)

        #TODO remove
        print("Global best:")
        print(best_particle_fitnes)
        print("Iteration:")
        print(iteration)
        print(global_best)

        iteration += 1

    print("Solution:")
    print(global_best)
    print(satisfied_clauses(global_best, clauses))
    print("In ", iteration, " iterations")


def run_brute_force(filename):
    clauses, literals = clauses_from_file(os.path.abspath(filename))
    return brute_force_algorithm(clauses, literals)

def run_random(filename, num_of_iters):
    clauses, literals = clauses_from_file(os.path.abspath(filename))
    return random_algorithm(clauses, literals, num_of_iters)

def run_simulated_annealing(filename, num_of_iters):
    clauses, literals = clauses_from_file(os.path.abspath(filename))
    return simulated_annealing_algorithm(clauses, literals, num_of_iters)




def main():
    # max, val_list = run_brute_force("examples/input_easy.cnf")
    # print(max, val_list)

    # max, val_list = run_simulated_annealing("examples/input_sudoku.cnf", 400)
    # print(max, val_list)

    clauses, literals = w_clauses_from_file(os.path.abspath("examples/f600.cnf"))
    PSO(clauses, literals, 20, w = 1, c1 = 2, c2 = 2,  max_iteration = 10)

    print("Random:")
    max, val_list = run_random("examples/f600.cnf", 400)
    print(max, val_list)

if __name__ == "__main__":
    main()
