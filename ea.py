#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from random import random, randint, uniform, sample, choice
import argparse


class EA:
    def __init__(self,
                path,
                max_iterations,
                generation_size,
                mutation_rate = None,
                reproduction_size = None,
                crossover_p = None,
                tournament_k = None,
                lambda_star = None,
                alpha = None,
                max_flip = None,
                max_tabu_size = None):

        #Read clauses with weights from file
        self.clauses, self.num_literals, self.num_clauses = w_clauses_from_file(os.path.abspath(path))

        self.max_iterations = max_iterations
        self.current_iteration = 1
        self.generation_size = generation_size
        self.mutation_rate = mutation_rate
        self.reproduction_size = reproduction_size
        self.crossover_p = crossover_p
        self.tournament_k = tournament_k
        #for SAWEA
        self.lambda_star = lambda_star
        #for RFEA
        self.alpha = alpha
        self.variables_weights = {i:0 for i in range(self.num_literals)}
        #for FlipGA
        self.max_flip = max_flip
        #for ASAP
        self.tabu_list = []
        self.max_tabu_size = max_tabu_size

        #Initialize population using random approach
        self.population = [[randint(0,1) for x in range(self.num_literals)] for y in range(self.generation_size)]
        self.top_chromosome = self.population[0]


    def is_clause_satisfied(self, valuation_list, clause):
        """
        Returns True if clause is true (satisfied) or False if not satisfied.
        """
        for literal in clause:
            if literal < 0:
                v = 1 - valuation_list[-literal - 1]
            else:
                v = valuation_list[literal - 1]
            if v == 1:
                return True
        return False


    def fitness(self, chromosome):
        """
        Number of satisfied clauses with given valuation
        """
        num_true_clauses = 0
        for c in self.clauses:
            num_true_clauses += self.is_clause_satisfied(chromosome, c['clause'])

        return num_true_clauses


    def stop_condition(self):
        return self.current_iteration > self.max_iterations or self.fitness(self.top_chromosome) == self.num_clauses


    def selectionTop10(self):
        """
        Perform selection using top 10 best fitness chromosome approach
        """
        sorted_chromos = sorted(self.population, key = lambda chromo: self.fitness(chromo), reverse=True)
        selected_chromos = sorted_chromos[:10]

        return selected_chromos


    def selection_tournament_pick_one(self):
        """
        Chooses one chromosome using tournament selection.
        Parameter k defines how many chromosomes we take from population
        """
        the_chosen_ones = []
        top_i = None

        #Choose k random chromosomes for battle from population and search for chromosome with
        #highest fitness
        for i in range(self.tournament_k):
            pick = randint(0, self.num_literals - 1)
            the_chosen_ones.append(self.population[i])
            if top_i == None or self.fitness(the_chosen_ones[i]) > self.fitness(the_chosen_ones[top_i]):
                top_i = i

        self.population.remove(the_chosen_ones[top_i])
        self.tournament_k -= 1
        return the_chosen_ones[top_i]


    def selectionTournament(self):
        """
        Function chooses self.reproduction_size chromosomes using tournament selection
        """
        selected = [self.selection_tournament_pick_one() for i in range(self.reproduction_size)]
        self.tournament_k += self.reproduction_size
        return selected


    def selection_roulette_pick_one(self, sum_fitness):
        """
        Chooses one chromosome using roulette selection.
        """
        pick = uniform(0, sum_fitness)
        value = 0
        i = 0

        for chromosome in self.population:
            i += 1
            value += self.fitness(chromosome)
            if value > pick:
                return chromosome


    def selectionRoulette(self):
        """
        Function chooses self.reproduction_size chromosomes using roulette selection
        """
        sum_fitness = sum(self.fitness(chromosome) for chromosome in self.population)
        return [self.selection_roulette_pick_one(sum_fitness) for i in range(self.reproduction_size)]


    def crossover(self, a, b):
        """
        Perform uniform crossover with given probability crossover_p
        """
        ab = a.copy()
        ba = b.copy()

        for i in range(len(a)):
            p = random()
            if p < self.crossover_p:
                ab[i] = a[i]
                ba[i] = b[i]
            else:
                ab[i] = b[i]
                ba[i] = a[i]

        return (ab,ba)


    def mutation(self, chromosome):
        """
        Performing mutation over chromosome with given probability mutation_rate
        """
        t = random()
        if t < self.mutation_rate:
            #We do mutation
            i = randint(0, self.num_literals-1)
            chromosome[i] = 1 - chromosome[i]


    def create_generation(self, for_reproduction):
        """
        With individuals obtained in for_reproduction array we generate new generation
        using genetic operators crossover and mutation. The new generation is the same
        length as at the starting point.
        """
        new_generation = []

        #While we don't fill up new_generation
        while len(new_generation) < self.generation_size:
            #Pick 2 randomly and do crossover
            parents = sample(for_reproduction, 2)
            child1, child2 = self.crossover(parents[0], parents[1])

            #Perform mutation after crossover
            self.mutation(child1)
            self.mutation(child2)

            #Add new chromosomes into new generation
            new_generation.append(child1)
            new_generation.append(child2)

        self.top_chromosome = max(new_generation, key = lambda chromo: self.fitness(chromo))
        return new_generation


    def fitness_SAW(self, chromosome):
        """
        Fitness function with weights
        in order to identify the hard clauses
        """
        return sum(map(lambda i: i['w'] * self.is_clause_satisfied(chromosome, i['clause']), self.clauses))


    def mutation_one(self, chromosome):
        i = randint(0, self.num_literals-1)
        chromosome[i] = 1 - chromosome[i]


    def lamarckian_mutation(self, chromosome):
        """
        Set of randomly chosen clauses is generated. If each clause in this set is
        satisfied by given chromosome, then do nothing. Otherwise pick a random variable of an
        unsatisfied clause and flip its corresponding bit such that it satisfies the clause.
        """

        #Generate set of randomly chosen clauses
        random_clauses = sample(self.clauses, randint(1, self.num_clauses))

        unsatisfied_clauses = self.get_unsatisfied_clauses(chromosome, random_clauses)
        unsat_clauses_size = len(unsatisfied_clauses)
        if unsat_clauses_size == 0:
            return

        chosen_unsatisfied_clause = choice(unsatisfied_clauses)

        while True:
            pos = randint(0, unsat_clauses_size - 1)
            var = abs(chosen_unsatisfied_clause[pos])
            chromosome[var-1] = 1 - chromosome[var-1]
            if self.is_clause_satisfied(chromosome, chosen_unsatisfied_clause):
                break
            else:
                chromosome[var-1] = 1 - chromosome[var-1]


    def create_generation_1_Lambda(self):
        """
        (1, lambda) - 1 parent reproducing lambda offspring. ',' denotes that the best individuals
        in offspring will form the next generation
        """
        parent = self.population[0]
        children = []

        #Choose lambda offspring
        i = 0
        while i < self.lambda_star:
            child = parent.copy()
            #TODO probati sa Lamarckian mutacijom - proveriti, jer pise da treba od skupa dece odabrati jedno i onda vrsiti mutaciju
            self.mutation_one(child)
            #self.lamarckian_mutation(child_for_mutation)

            if child not in children:
                children.append(child)
                i += 1

        best = max(children, key = lambda chromo: self.fitness_SAW(chromo))
        self.top_chromosome = best
        return [best]


    def update_clauses_weight(self):
        """
        Update clauses weight to identify the hard clauses
        """
        for clause in self.clauses:
            clause['w'] = clause['w'] + 1 - self.is_clause_satisfied(self.top_chromosome, clause['clause'])


    def K(self, x):
        return -1 if x == 0 else 1


    def update_v(self, i):
        """
        Update weigths of variables
        """
        sum = 0

        #Summing weights of unsatisfied clauses containing variable i
        for c in self.clauses:
            if(self.is_clause_satisfied(self.top_chromosome, c['clause']) == False and (i+1) in c['clause']):
                sum += c['w']

        self.variables_weights[i] = self.variables_weights[i] - self.K(self.top_chromosome[i])*sum


    def r(self, chromosome):
        """
        Function used in fitness_REF
        """
        literals = range(self.num_literals) #list [1,2,3..., num_literals]

        arg_up = sum(map(lambda i: self.K(chromosome[i]) * self.variables_weights[i], literals))
        arg_down = 1 + sum(map(lambda i: abs(self.variables_weights[i]), literals))

        for i in literals:
            self.update_v(i)

        return 0.5*(1 + arg_up/arg_down)


    def fitness_REF(self, chromosome):
        return sum(map(lambda i: self.is_clause_satisfied(chromosome, i['clause']), self.clauses)) + self.alpha*self.r(chromosome)


    def get_unsatisfied_clauses(self, chromosome, clauses):
        unsatisfied = []

        for i in clauses:
            if self.is_clause_satisfied(chromosome, i['clause']) == False:
                unsatisfied.append(i['clause'])

        return unsatisfied


    def mutation_knowledge_based(self, chromosome):
        """
        Selects an unsat­isfied clause and flips exactly one randomly chosen variable contained in the clause
        """
        unsatisfied_clauses = self.get_unsatisfied_clauses(chromosome, self.clauses)
        chosen_unsatisfied_clause = choice(unsatisfied_clauses)
        pos = randint(0, len(chosen_unsatisfied_clause)-1)
        var = abs(chosen_unsatisfied_clause[pos])
        chromosome[var-1] = 1 - chromosome[var-1]


    def create_generation_steady_state(self, for_reproduction):
        """
        Steady state replacement eliminating the worst individual
        """

        parent1, parent2 = for_reproduction[0], for_reproduction[1]
        child1 = parent1.copy()
        child2 = parent2.copy()

        #no crossover!
        best1, best2 = None, None

        while best1 == best2:

            #TODO probati sa Lamarckian SEA-SAW mutation operator
            # self.mutation_knowledge_based(child1)
            # self.mutation_knowledge_based(child2)
            self.lamarckian_mutation(child1)
            self.lamarckian_mutation(child2)

            x = [parent1, parent2, child1, child2]

            #2 highest fitness individuals out of parent1, parent2, child1, child2
            best1, best2 = sorted(x, key = lambda chromo: self.fitness_REF(chromo), reverse = True)[:2]

        self.population.append(best1.copy())
        self.population.append(best2.copy())

        self.top_chromosome = max(self.population, key = lambda chromo: self.fitness_REF(chromo))


    def local_search(self, chromosome):
        improvement = 1
        nbrflip = 0

        while(improvement > 0 and nbrflip < self.max_flip):

            improvement = 0
            for i in range(self.num_literals):
                fit_before = self.fitness(chromosome)
                #Flip the i-th variable of the particle
                chromosome[i] = 1 - chromosome[i]
                nbrflip += 1
                fit_after = self.fitness(chromosome)

                gain = fit_after - fit_before
                if gain >= 0:
                    #Accept flip
                    improvement += gain
                else:
                    #There is no improvement
                    #Undo flip
                    chromosome[i] = 1 - chromosome[i]


    def create_generation_generational(self, for_reproduction):
        new_generation = []

        #Pick 2 best parents and do crossover
        parents = sorted(for_reproduction, key = lambda chromo: self.fitness(chromo), reverse=True)[:2]

        #Adding to new generation best of current generation
        new_generation.append(parents[0])
        new_generation.append(parents[1])

        #While we don't fill up new_generation
        while len(new_generation) < self.generation_size:
            child1, child2 = self.crossover(parents[0], parents[1])

            #Perform mutation after crossover
            self.mutation(child1)
            self.mutation(child2)

            self.local_search(child1)
            self.local_search(child2)

            #Add new chromosomes into new generation
            new_generation.append(child1)
            new_generation.append(child2)

        self.top_chromosome = max(new_generation, key = lambda chromo: self.fitness(chromo))
        return new_generation


    def local_search_tabu(self, chromosome):
        improvement = 1
        nbrflip = 0

        while(  improvement > 0
                and nbrflip < self.max_flip
                and chromosome not in self.tabu_list):

            improvement = 0
            for i in range(self.num_literals):
                fit_before = self.fitness(chromosome)
                #Flip the i-th variable of the particle
                chromosome[i] = 1 - chromosome[i]
                nbrflip += 1
                fit_after = self.fitness(chromosome)

                gain = fit_after - fit_before
                if gain >= 0:
                    #Accept flip
                    improvement += gain
                else:
                    #There is no improvement
                    #Undo flip
                    chromosome[i] = 1 - chromosome[i]

            if improvement == 0:
                #If no improvement add this solution to tabu list
                self.tabu_list.append(chromosome)
                if len(self.tabu_list) > self.max_tabu_size:
                    del self.tabu_list[0]


    def create_generation_1_plus_1(self):
        """
        (1+1) - 1 parent reproducing 1 child. '+' denotes elitism strategy on both generations
        """
        parent = self.population[0]
        child = parent.copy()

        #TODO random-adaptive mutation
        self.mutation_one(child)
        self.local_search_tabu(child)

        best = child if self.fitness(child) > self.fitness(parent) else parent
        self.top_chromosome = best
        return [best]


def run(path):
    ea = EA(path)

    #While stop condition is not archieved
    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        #From population choose chromosomes for reproduction

        # for_reproduction = ea.selectionTop10()
        # for_reproduction = ea.selectionTournament()
        for_reproduction = ea.selectionRoulette()

        #Show current state of algorithm
        print('Current solution fitness = %d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.population = ea.create_generation(for_reproduction)

        ea.current_iteration += 1

    #Return best chromosome in the last population
    return (ea.top_chromosome, ea.fitness(ea.top_chromosome))


def run_SAWEA(path, max_iterations, lambda_star):
    """
    Using stepwise adaption of weights
    """
    ea = EA(
        path,
        max_iterations,
        generation_size = 1,
        lambda_star = lambda_star
    )

    #While stop condition is not archieved
    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        #Show current state of algorithm
        print('Current solution SAW fitness = %d' % ea.fitness_SAW(ea.top_chromosome))
        print('Current solution fitness = %d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.population = ea.create_generation_1_Lambda()

        ea.current_iteration += 1

        ea.update_clauses_weight()

    return (ea.top_chromosome, ea.fitness(ea.top_chromosome), ea.current_iteration)


def run_RFEA(path, max_iterations, crossover_p, alpha):

    ea = EA(
        path,
        max_iterations,
        generation_size = 4,
        reproduction_size = 2,
        crossover_p = crossover_p,
        tournament_k = 4,
        alpha = alpha)

    #While stop condition is not archieved
    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        #From population choose chromosomes for reproduction
        for_reproduction = ea.selectionTournament()

        #Show current state of algorithm
        print('Current solution RFEA fitness = %d' % ea.fitness_REF(ea.top_chromosome))
        print('Current solution fitness = %d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.create_generation_steady_state(for_reproduction)

        ea.current_iteration += 1

        ea.update_clauses_weight()

    #Return best chromosome in the last population
    return (ea.top_chromosome, ea.fitness(ea.top_chromosome), ea.current_iteration)


def run_FlipGA(path, max_iterations, crossover_p, max_flip):

    ea = EA(
        path,
        max_iterations,
        generation_size = 10,
        reproduction_size = 5,
        mutation_rate = 0.9,
        crossover_p = crossover_p,
        max_flip = max_flip)

    #While stop condition is not archieved
    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        #From population choose chromosomes for reproduction
        for_reproduction = ea.selectionRoulette()

        #Show current state of algorithm
        print('Current solution fitness = %d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.population = ea.create_generation_generational(for_reproduction)

        ea.current_iteration += 1

    #Return best chromosome in the last population
    return (ea.top_chromosome, ea.fitness(ea.top_chromosome), ea.current_iteration)


def run_ASAP(path, max_iterations, max_flip, max_tabu_size):

    ea = EA(
        path,
        max_iterations,
        generation_size = 1,
        mutation_rate = 0.9,
        max_flip = max_flip,
        max_tabu_size = max_tabu_size)

    #While stop condition is not archieved
    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        #Show current state of algorithm
        print('Current solution fitness = %d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.population = ea.create_generation_1_plus_1()

        ea.current_iteration += 1

    #Return best chromosome in the last population
    return (ea.top_chromosome, ea.fitness(ea.top_chromosome), ea.current_iteration)


def w_clauses_from_file(filename):
    """
    Returns array of clauses with weights = 1
    [{'clause':[1,2,3], 'w':1}, {'clause':[-1,2], 'w':1}... ]
    and number of literals
    """
    clauses_with_weights = []

    with open(filename, "r") as fin:
        #remove comments from beginning
        line = fin.readline()
        while(line.lstrip()[0] == 'c'):
            line = fin.readline()

        header = line.split(" ")
        num_literals = int(header[2].rstrip())
        num_clauses = int(header[3].rstrip())

        lines = fin.readlines()
        for line in lines:
            line = line.split(" ")[:-1]
            line = [int(x) for x in line]
            clauses_with_weights.append({'clause':line, 'w':1})

        return (clauses_with_weights, num_literals, num_clauses)


def main():
    #Parsing arguments of command line
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = "path to input .cnf file")
    parser.add_argument('algorithm',
                        choices=['sawea','rfea','flipga', 'asap'],
                        help = "Choose an algorithm to run")
    parser.add_argument('-i', '--max_iterations',
                        nargs = '?', default = 1000, type = int,
                        help = "Maximal number of iterations. Default 1000")
    parser.add_argument('-g', '--generation_size',
                        nargs = '?', default = 100, type = int,
                        help = "Generation size. Default 100")
    parser.add_argument('-m', '--mutation_rate',
                        nargs = '?', default = 0.01, type = float,
                        help = "Mutation probability.  Default 0.01")
    parser.add_argument('-r', '--reproduction_size',
                        nargs = '?', default = 10, type = int,
                        help = "Default 10")
    parser.add_argument('-p', '--crossover',
                        nargs = '?', default = 0.5, type = float,
                        help = "Crossover probability. Default 0.5")
    parser.add_argument('-k', '--tournament_k',
                        nargs = '?', default = 3, type = int,
                        help = "Number of elements in tournament selection. Default 3")
    parser.add_argument('-l', '--lambda_star',
                        nargs = '?', default = 10, type = int,
                        help = "lambda* for SAWEA. Default 10")
    parser.add_argument('-a', '--alpha',
                        nargs = '?', default = 0.5, type = float,
                        help = "alpha for RFEA. Default 0.5")
    parser.add_argument('-f', '--max_flip',
                        nargs = '?', default = 30000, type = int,
                        help = "Maximal number of flips for FlipGA. Default 30000")
    parser.add_argument('-t', '--max_tabu_size',
                        nargs = '?', default = 5, type = int,
                        help = "Maximal number of elements in tabu list. Default 5")
    args = parser.parse_args()

    #run("examples/aim-50-2_0-yes.cnf")

    if (args.algorithm == "sawea"):
        print("SAWEA".center(40, "-"))
        solution, fitness, iteration = run_SAWEA(
            path                = args.path,
            max_iterations      = args.max_iterations,
            lambda_star         = args.lambda_star
        )

    if (args.algorithm == "rfea"):
        print("RFEA".center(40, "-"))
        solution, fitness, iteration = run_RFEA(
            path                = args.path,
            max_iterations      = args.max_iterations,
            crossover_p         = args.crossover,
            alpha               = args.alpha
        )

    if (args.algorithm == "flipga"):
        print("FlipGA".center(40, "-"))
        solution, fitness, iteration = run_FlipGA(
            path                = args.path,
            max_iterations      = args.max_iterations,
            crossover_p         = args.crossover,
            max_flip            = args.max_flip
        )

    if (args.algorithm == "asap"):
        print("ASAP".center(40, "-"))
        solution, fitness, iteration = run_ASAP(
            path                = args.path,
            max_iterations      = args.max_iterations,
            max_flip            = args.max_flip,
            max_tabu_size       = args.max_tabu_size
        )


    print("Solution:")
    print(solution)
    print("Satisfied clauses: ", fitness)
    print("In ", iteration, " iterations")



if __name__ == "__main__":
    main()
