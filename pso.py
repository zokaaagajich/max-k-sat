#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randint, uniform, random
from math import exp


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
        r1 = random()
        r2 = random()
        new_velocity = []
        for i in range(self.num_literals):
            new_velocity.append( w*self.velocity[i] + c1*r1*(self.best[i] - self.position[i]) + c2*r2*(global_best[i] - self.position[i]) )
        self.velocity = new_velocity


    def sigmoid(self, velocity):
        return 1.0/(1 + exp(-velocity))


    def update_position(self):
        """
        Update the particle position
        """
        #TODO using flight
        r = random()
        new_position = []
        for i in range(self.num_literals):
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


    def init_particles(self, num_particles, particle_size):
        swarm = []
        for i in range(num_particles):
            swarm.append(Particle(particle_size))
        return swarm


    def __init__(self, filename, num_particles, max_iteration, w = 1, c1 = 2, c2 = 2):
        """
        Read cnf from file and
        Initialize the parameters, population, positions and velocities
        """
        #Read cnf formula from file
        self.clauses, self.num_literals = self.w_clauses_from_file(filename)

        #Parameters of PSO
        self.num_particles = num_particles
        self.max_iteration = max_iteration
        self.w = w
        self.c1 = c1
        self.c2 = c2

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


    def calc_fitness_and_global_best(self):
        for particle in self.swarm:
            #Evaluate the fitness of the each particle (Pi)
            particle.fitness = self.fitness(particle.position)

            #Save the individuals highest fitness (Pg)
            #Update global best
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best = particle.position


    def update_velocities(self, particle):
        particle.update_velocity(self.global_best, self.w, self.c1, self.c2)


    def update_positions(self, particle):
        particle.update_position()


    def update_personal_best(self, particle):
        p_best_fit = self.fitness(particle.best)
        if particle.fitness > p_best_fit:
            particle.fitness = p_best_fit
            particle.best = particle.position


    def update_global_best(self, particle):
        curr_fitness = self.fitness(particle.best)
        if curr_fitness > self.global_best_fitness:
            self.global_best_fitness = curr_fitness
            self.global_best = particle.best


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


    def stop_condition(self, iteration):
        if self.num_satisfied_clauses(self.global_best) == self.num_literals or iteration > self.max_iteration:
            return True
        return False


def main():
    pso = PSO("../max-k-sat/examples/f600.cnf", num_particles = 20, max_iteration = 100, w = 1, c1 = 2, c2 = 2)

    iteration = 1
    while(not pso.stop_condition(iteration)):
        #Calculate fitness
        #Save the individuals highest fitness (Pi)
        #Update global best
        pso.calc_fitness_and_global_best()

        for particle in pso.swarm:
            #Modify velocities
            pso.update_velocities(particle)

            #Update the particles position
            #TODO using flight
            pso.update_positions(particle)

            #Update particle best
            pso.update_personal_best(particle)

            #Update global best
            pso.update_global_best(particle)

        pso.update_clauses_weight()

        #TODO remove
        print("Iteration: ", iteration)
        print("Global best fitness: ", pso.global_best_fitness)

        iteration += 1

    print("Solution:")
    print(pso.global_best)
    print(pso.num_satisfied_clauses(pso.global_best))
    print("In ", iteration, " iterations")


if __name__ == "__main__":
    main()
