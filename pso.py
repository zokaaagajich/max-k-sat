#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randint, uniform, random, sample
from math import exp
import argparse

class Particle:
    """
    Particle
    """

    def __init__(self, num_literals):
        """
        Initialize particle position, velocity and personal best solution
        """
        self.num_literals = num_literals
        self.position = [randint(0,1) for x in range(num_literals)]
        self.best = self.position
        #initial velocities from -1 to 1
        self.velocity = [2*random()-1 for x in range(num_literals)]
        self.fitness = float("-inf")


    def update_velocity(self, global_best, w, c1, c2):
        """
        Update particle velocity
        """
        new_velocity = []
        for i in range(self.num_literals):
            r1 = random()
            r2 = random()
            new_velocity.append( w*self.velocity[i] + c1*r1*(self.best[i] - self.position[i]) + c2*r2*(global_best[i] - self.position[i]) )
        self.velocity = new_velocity


    def sigmoid(self, velocity):
        return 1.0/(1 + exp(-velocity))


    def update_position(self):
        """
        Update the particle position
        """
        new_position = []
        for i in range(self.num_literals):
            r = random()
            position_i = 1 if r < self.sigmoid(self.velocity[i]) else 0
            new_position.append(position_i)
        self.position = new_position


    def __str__(self):
        return "Position:\n" + str(self.position) + "\nVelocity:\n" + str(self.velocity) + "\nPersonal best:\n" + str(self.best)



class PSO:
    """
    Particle Swarm Optimization
    """

    def w_clauses_from_file(self, filename):
        """
        Returns array of clauses with weights = 1
        [{'clause':[1,2,3], 'w':1}, {'clause':[-1,2], 'w':1}... ]
        and number of literals
        """
        clauses = []
        with open(filename, "r") as fin:
            #remove comments from beginning
            line = fin.readline()
            while line.lstrip()[0] == 'c':
                line = fin.readline()

            header = line.split(" ")
            num_literals = int(header[2].rstrip())
            num_clauses = int(header[3].rstrip())


            lines = fin.readlines()
            for line in lines:
                line = line.split(" ")[:-1]
                line = [int(x) for x in line]
                clauses.append({'clause':line, 'w':1})

            return (clauses, num_literals, num_clauses)


    def init_particles(self, num_particles, particle_size):
        swarm = []
        for i in range(num_particles):
            swarm.append(Particle(particle_size))
        return swarm


    def __init__(self, filename, num_particles, max_iteration, maxFlip, maxTabuSize, w, c1, c2):
        """
        Read cnf from file and
        Initialize the parameters, population, positions and velocities
        """
        #Read cnf formula from file
        self.clauses, self.num_literals, self.num_clauses = self.w_clauses_from_file(filename)

        #Parameters of PSO
        self.num_particles = num_particles
        self.max_iteration = max_iteration
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_flip = maxFlip

        #Tabu list parameters
        self.tabuList = []
        self.maxTabuSize = maxTabuSize

        #Initialize particles
        self.swarm = self.init_particles(self.num_particles, self.num_literals)

        #Initialize global best and it's fitness
        self.global_best = self.swarm[0].position
        self.global_best_fitness = self.fitness(self.global_best)


    def __str__(self):
        _str = ""
        for i, particle in enumerate(self.swarm):
            _str += "Particle " + str(i) + ":\n" + str(particle) + "\n"
        return _str


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


    def fitness(self, valuation):
        """
        Fitness function with weights
        in order to identify the hard clauses
        """
        return sum(map(lambda i: i['w'] * self.is_clause_satisfied(valuation, i['clause']), self.clauses))


    def calc_fitness_and_global_best(self, fitness):
        for particle in self.swarm:
            #Evaluate the fitness of the each particle (Pi)
            particle.fitness = fitness(particle.position)

            #Save the individuals highest fitness (Pg)
            #Update global best
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best = particle.position


    def update_velocities(self, particle):
        particle.update_velocity(self.global_best, self.w, self.c1, self.c2)


    def update_positions(self, particle):
        particle.update_position()


    def update_personal_best(self, particle, fitness):
        p_best_fit = fitness(particle.best)
        if particle.fitness > p_best_fit:
            particle.fitness = p_best_fit
            particle.best = particle.position


    def update_global_best(self, particle, fitness):
        curr_fitness = fitness(particle.best)
        if curr_fitness > self.global_best_fitness:
            self.global_best_fitness = curr_fitness
            self.global_best = particle.best


    def update_global_best_ring(self, particle, neighbor_particle):
        fitness_p = self.fitness(particle.best)
        fitness_n = self.fitness(neighbor_particle.best)
        (self.global_best, self.global_best_fitness) = (particle.best, fitness_p) if fitness_p > fitness_n else (neighbor_particle.best, fitness_n)


    def update_clauses_weight(self):
        """
        Update clauses weight to identify the hard clauses
        """
        for clause in self.clauses:
            clause['w'] = clause['w'] + 1 - self.is_clause_satisfied(self.global_best, clause['clause'])


    def num_satisfied_clauses(self, val_list):
        """
        Number of satisfied clauses with given valuation
        """
        num_true_clauses = 0
        for c in self.clauses:
            num_true_clauses += self.is_clause_satisfied(val_list, c['clause'])
        return num_true_clauses


    def stop_condition(self, num_satisfied_clauses, iteration):
        return iteration >= self.max_iteration or num_satisfied_clauses == self.num_clauses


    def local_search(self, particle, fitness):
        improvement = 1
        nbrflip = 0

        while  improvement > 0 and nbrflip < self.max_flip:
            improvement = 0
            for i in range(self.num_literals):
                fit_before = fitness(particle.position)
                #Flip the i-th variable of the particle
                particle.position[i] = 1 - particle.position[i]
                nbrflip += 1
                fit_after = fitness(particle.position)

                gain = fit_after - fit_before
                if gain >= 0:
                    #Accept flip
                    improvement += gain
                else:
                    #There is no improvement
                    #Undo flip
                    particle.position[i] = 1 - particle.position[i]


    def local_search_random_k(self, particle, fitness, k):
        """
        Local search that doesn't flip every bit of particle,
        but only randomly chosen k % of particle length
        """
        improvement = 1
        nbrflip = 0

        while  improvement > 0 and nbrflip < self.max_flip and particle.position not in self.tabuList:
            improvement = 0
            for i in sample(range(self.num_literals), int(self.num_literals*k)):
                fit_before = fitness(particle.position)
                #Flip the i-th variable of the particle
                particle.position[i] = 1 - particle.position[i]
                nbrflip += 1
                fit_after = fitness(particle.position)

                gain = fit_after - fit_before
                if gain >= 0:
                    #Accept flip
                    improvement += gain
                else:
                    #There is no improvement
                    #Undo flip
                    particle.position[i] = 1 - particle.position[i]

            if improvement == 0:
                #If no improvement add this solution to tabu list
                self.tabuList.append(particle.position)
                if len(self.tabuList) > self.maxTabuSize:
                    del self.tabuList[0]


def run_PSO_LS(path, num_particles, max_iteration, max_flip, w, c1, c2):
    """
    PSO with flight operation based on sigmoid transformation
    """
    pso = PSO(path, num_particles, max_iteration, max_flip, 0, w, c1, c2)
    iteration = 0
    num_satisfied_clauses = 0

    while not pso.stop_condition(num_satisfied_clauses, iteration):
        iteration += 1

        #Calculate fitness
        #Save the individuals highest fitness (Pi)
        #Update global best
        pso.calc_fitness_and_global_best(pso.fitness)

        for i, particle in enumerate(pso.swarm):
            #Modify velocities
            pso.update_velocities(particle)

            #Update the particles position
            pso.update_positions(particle)

            #Update particle best
            pso.update_personal_best(particle, pso.fitness)

            #Update global best
            pso.update_global_best(particle, pso.fitness)
            #Topology ring size 1
            #pso.update_global_best_ring(particle, pso.swarm[i % (pso.num_particles+1)])

        if iteration % 10 == 0:
            pso.update_clauses_weight()

        print("Iteration: ", iteration)
        num_satisfied_clauses = pso.num_satisfied_clauses(pso.global_best)
        print("Satisfied clauses:")
        print(num_satisfied_clauses)
        print("Fitness: ", pso.global_best_fitness)

    return (pso.global_best, num_satisfied_clauses, iteration)


def run_PSOSAT(path, num_particles, max_iteration, max_flip, maxTabuSize, w, c1, c2):
    """
    PSO with the standard objective function - number of satisfied clauses
    """
    pso = PSO(path, num_particles, max_iteration, max_flip, maxTabuSize, w, c1, c2)
    iteration = 0

    while not pso.stop_condition(pso.global_best_fitness, iteration):
        iteration += 1

        #Calculate fitness
        #Save the individuals highest fitness (Pi)
        #Update global best
        pso.calc_fitness_and_global_best(pso.num_satisfied_clauses)

        for i, particle in enumerate(pso.swarm):
            #Update the particles position
            pso.local_search(particle, pso.num_satisfied_clauses)

            #Update particle best
            pso.update_personal_best(particle, pso.num_satisfied_clauses)

            #Update global best
            pso.update_global_best(particle, pso.num_satisfied_clauses)
            #Topology ring size 1
            #pso.update_global_best_ring(particle, pso.swarm[i % (pso.num_particles+1)])

        print("Iteration: ", iteration)
        print("Satisfied clauses:")
        print(pso.global_best_fitness)

    return (pso.global_best, pso.global_best_fitness, iteration)


def run_WPSOSAT(path, num_particles, max_iteration, max_flip, maxTabuSize, w, c1, c2):
    pso = PSO(path, num_particles, max_iteration, max_flip, maxTabuSize, w, c1, c2)
    iteration = 0
    num_satisfied_clauses = 0

    while not pso.stop_condition(num_satisfied_clauses, iteration):
        iteration += 1

        #Calculate fitness
        #Save the individuals highest fitness (Pi)
        #Update global best
        pso.calc_fitness_and_global_best(pso.fitness)

        for i, particle in enumerate(pso.swarm):
            #Update the particles position
            pso.local_search(particle, pso.fitness)

            #Update particle best
            pso.update_personal_best(particle, pso.fitness)

            #Update global best
            pso.update_global_best(particle, pso.fitness)
            #Topology ring size 1
            #pso.update_global_best_ring(particle, pso.swarm[i % (pso.num_particles+1)])

        if iteration % 10 == 0:
            pso.update_clauses_weight()

        print("Iteration: ", iteration)
        num_satisfied_clauses = pso.num_satisfied_clauses(pso.global_best)
        print("Satisfied clauses:")
        print(num_satisfied_clauses)
        print("Fitness: ", pso.global_best_fitness)

    return (pso.global_best, num_satisfied_clauses, iteration)



def main():
    #parsing arguments of command line
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = "path to input .cnf file")
    parser.add_argument('algorithm', choices=['psols','psosat','wpsosat'], help = "Choose an algorithm to run")
    parser.add_argument('-p', '--particles', nargs = '?', default = 20, type = int, help = "number of particles in generation. Default 20")
    parser.add_argument('-i', '--maxIter', nargs = '?', default = 1000, type = int, help = "maximal number of generations. Default 1000")
    parser.add_argument('-t', '--maxTabuSize', nargs = '?', default = 500, type = int, help = "maximal number of elements in tabu list. Default 500")
    parser.add_argument('-f', '--maxFlip', nargs = '?', default = 30000, type = int, help = "maximal number of flips in flip heuristic. Default 30000")
    parser.add_argument('-w', '--inertia', nargs = '?', default = 1, type = float, help = "inertia factor. Default 1")
    parser.add_argument('-c1', '--individual', nargs = '?', default = 1.7, type = float, help = "individual factor. Default 1.7")
    parser.add_argument('-c2', '--social', nargs = '?', default = 2.1, type = float, help = "social factor. Default 2.1")
    args = parser.parse_args()


    if args.algorithm == "psols":
        solution, satisfied_clauses, iteration = run_PSO_LS(path = args.path,
                num_particles = args.particles,
                max_iteration = args.maxIter,
                max_flip = args.maxFlip,
                w = args.inertia,
                c1 = args.individual,
                c2 = args.social)

    if args.algorithm == "psosat":
        solution, satisfied_clauses, iteration = run_PSOSAT(path = args.path,
                num_particles = args.particles,
                max_iteration = args.maxIter,
                max_flip = args.maxFlip,
                maxTabuSize = args.maxTabuSize,
                w = args.inertia,
                c1 = args.individual,
                c2 = args.social)

    if args.algorithm == "wpsosat":
        solution, satisfied_clauses, iteration = run_WPSOSAT(path = args.path,
                num_particles = args.particles,
                max_iteration = args.maxIter,
                max_flip = args.maxFlip,
                maxTabuSize = args.maxTabuSize,
                w = args.inertia,
                c1 = args.individual,
                c2 = args.social)

    print("Solution:")
    print(solution)
    print("Satisfied clauses: ", satisfied_clauses)
    print("In ", iteration, " iterations")


if __name__ == "__main__":
    main()
