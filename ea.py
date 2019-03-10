#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import operator
import random
import math

class EA:

    def __init__(self, path,
                max_iterations = 1000,
                generation_size = 100,
                mutation_rate = 0.01,
                reproduction_size = 10,
                crossover_p = 0.5,
                tournament_k = 3,
                lambda_star = 10,
                alpha = 0.5,
                max_flip = 30000):

        self.clauses, self.num_literals, self.num_clauses = w_clauses_from_file(os.path.abspath(path))
        self.max_iterations = max_iterations
        self.generation_size = generation_size
        self.mutation_rate = mutation_rate
        self.reproduction_size = reproduction_size
        self.crossover_p = crossover_p
        self.tournament_k = tournament_k
        self.lambda_star = lambda_star
        self.alpha = alpha
        self.max_flip = max_flip

        self.current_iteration = 0
        self.variables_weights = {i:0 for i in range(self.num_literals)}

        #Initialize population using random approach
        self.population = [[random.randint(0,1) for x in range(self.num_literals)] for y in range(self.generation_size)]
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


    def goal_function(self, fitness, chromosome):
        """
        In case we find optimal solution, we remember gene as optimal and stop the
        algorithm (stop condition will be archieved).
        """

        if fitness == self.num_clauses:
            self.top_chromosome = chromosome


    def fitness(self, chromosome):
        """
        Number of satisfied clauses with given valuation
        """
        if chromosome == None:
            return 0

        num_true_clauses = 0
        for c in self.clauses:
            num_true_clauses += self.is_clause_satisfied(chromosome, c['clause'])

        self.goal_function(num_true_clauses, chromosome)

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


    def selection_tournament_pick_one(self, k):
        """
        Chooses one chromosome using tournament selection.
        Parameter k defines how many chromosomes we take from population
        """
        the_chosen_ones = []
        top_i = None

        #Choose k random chromosomes from population and search for chromosome with
        #highest fitness
        for i in range(k):
            pick = random.randint(0, self.num_literals - 1)
            the_chosen_ones.append(self.population[i])
            if top_i == None or self.fitness(the_chosen_ones[i]) > self.fitness(the_chosen_ones[top_i]):
                top_i = i

        return the_chosen_ones[top_i]


    def selectionTournament(self):
        """
        Function chooses self.reproduction_size chromosomes using tournament selection
        """
        selected_chromos = []
        selected_chromos = [self.selection_tournament_pick_one(self.tournament_k) for i in range(self.reproduction_size)]

        return selected_chromos


    def selection_roulette_pick_one(self, sum_fitness):
        """
        Chooses one chromosome using roulette selection.
        """
        pick = random.uniform(0, sum_fitness)
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
        selected_chromos = []
        selected_chromos = [self.selection_roulette_pick_one(sum_fitness) for i in range(self.reproduction_size)]

        return selected_chromos


    def crossover(self, a, b):
        """
        Perform uniform crossover with given probability crossover_p
        """
        ab = a.copy()
        ba = b.copy()

        for i in range(len(a)):
            p = random.random()
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
        t = random.random()
        if t < self.mutation_rate:
            #We do mutation
            i = random.randint(0, self.num_literals-1)
            chromosome[i] = 1 - chromosome[i]

        return chromosome


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
            parents = random.sample(for_reproduction, 2)
            child1, child2 = self.crossover(parents[0], parents[1])

            #Perform mutation after crossover
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

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
        ch_copy = chromosome.copy()
        i = random.randint(0, self.num_literals-1)
        ch_copy[i] = 1 - ch_copy[i]
        return ch_copy


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
            child = self.mutation_one(parent)
            if child not in children:
                children.append(child)
                i+=1

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


    def create_generation_steady_state(self, for_reproduction):
        """
        Steady state replacement eliminating the worst individual
        """
        new_generation = []

        #TODO sample(for_reproduction, 2)
        parent1, parent2 = for_reproduction[0], for_reproduction[1]
        #TODO
        child1, child2 = self.crossover(parent1, parent2)

        best1, best2 = None, None

        while best1 == best2:
            #TODO change mutation function! Lamarckian SEA-SAW mutation operator
            child1 = self.mutation_one(child1)
            child2 = self.mutation_one(child2)

            x = [parent1, parent2, child1, child2]

            #2 highest fitness individuals out of parent1, parent2, child1, child2
            best1, best2 = sorted(x, key = lambda chromo: self.fitness_REF(chromo), reverse = True)[:2]

        new_generation.append(best1)
        new_generation.append(best2)

        self.top_chromosome = max(new_generation, key = lambda chromo: self.fitness_REF(chromo))

        return new_generation


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
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            self.local_search(child1)
            self.local_search(child2)

            #Add new chromosomes into new generation
            new_generation.append(child1)
            new_generation.append(child2)

        self.top_chromosome = max(new_generation, key = lambda chromo: self.fitness(chromo))
        return new_generation


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


def run_SAWEA(path):
    """
    Using stepwise adaption of weights
    """
    ea = EA(path,
            generation_size = 1,
            reproduction_size = 10,
            max_iterations = 1000)

    #While stop condition is not archieved
    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        #Show current state of algorithm
        print('Current solution fitness = %d' % ea.fitness_SAW(ea.top_chromosome))
        print('Current solution fitness = %d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.population = ea.create_generation_1_Lambda()

        ea.current_iteration += 1

        ea.update_clauses_weight()

    return (ea.top_chromosome, ea.fitness(ea.top_chromosome))


def run_RFEA(path):
    #TODO why bug
    ea = EA(path,
            generation_size = 4,
            reproduction_size = 2,
            tournament_k = 2)

    #While stop condition is not archieved
    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        #From population choose chromosomes for reproduction
        for_reproduction = ea.selectionTournament()

        #Show current state of algorithm
        print('Current solution fitness = %d' % ea.fitness_REF(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.population = ea.create_generation_steady_state(for_reproduction)

        ea.current_iteration += 1

        ea.update_clauses_weight()

    #Return best chromosome in the last population
    return (ea.top_chromosome, ea.fitness(ea.top_chromosome))


def run_FlipGA(path):
    ea = EA(path,
            generation_size = 10,
            max_iterations = 100,
            mutation_rate = 0.9)

    #While stop condition is not archieved
    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        #From population choose chromosomes for reproduction
        for_reproduction = ea.selectionTop10()

        #Show current state of algorithm
        print('Current solution fitness = %d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.population = ea.create_generation_generational(for_reproduction)

        ea.current_iteration += 1

    #Return best chromosome in the last population
    return (ea.top_chromosome, ea.fitness(ea.top_chromosome))


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
    # TODO arguments!
    # run("examples/aim-50-2_0-yes.cnf")

    # run_SAWEA("examples/aim-50-2_0-yes.cnf")
    # run_RFEA("examples/aim-50-2_0-yes.cnf")
    solution, fitness = run_FlipGA("examples/aim-200-6_0-yes.cnf")
    print(solution, fitness)

if __name__ == "__main__":
    main()
