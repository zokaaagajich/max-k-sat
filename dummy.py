#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randint, uniform, random
import os
import math
import argparse


def clauses_from_file(filename):
    """
    Functions returns array of clauses [[1,2,3], [-1,2], ... ]
    and number of literals(variables)
    """
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


def binary_list(dec_number, width):
    """
    Returns binary representation of decimal number with specified width
    example: dec_number = 2 and width = 3, returns: 010
    """
    bin_str = bin(dec_number)[2:].zfill(width)
    return [int(x) for x in bin_str]


def is_clause_satisfied(valuation_list, clause):
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


def solution(i, literals, clauses):
    """
    For i combination returns valuation list and number of true clauses
    """
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


def brute_force_algorithm(clauses, literals):
    """
    For each combination from 0 to 2^num_of_vars count
    number of true clauses
    """
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


def random_algorithm(clauses, literals, num_of_iters):
    """
    For given number of iterations gives random number
    for which we check number of true clauses.
    Does not garantee best solution!
    """
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


def run_brute_force(filename):
    clauses, literals = clauses_from_file(os.path.abspath(filename))
    return brute_force_algorithm(clauses, literals)


def run_random(filename, num_of_iters):
    clauses, literals = clauses_from_file(os.path.abspath(filename))
    return random_algorithm(clauses, literals, num_of_iters)


def main():
    #parsing arguments of command line
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = "path to input .cnf file")
    parser.add_argument('algorithm', choices=['brute_force','random'], help = "Choose an algorithm to run")
    parser.add_argument('-i', '--iterations', nargs = '?', default = 500, type = int, help = "Number of iterations. Default 500")
    args = parser.parse_args()

    if (args.algorithm == "brute_force"):
        max, val_list = run_brute_force(args.path)

    if (args.algorithm == "random"):
        max, val_list = run_random(args.path, args.iterations)

    print("Solution:")
    print(val_list)
    print("Satisfied clauses: ", max)


if __name__ == "__main__":
    main()
