#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import argparse
import os

#Parsing arguments of command line
parser = argparse.ArgumentParser()
parser.add_argument('path', help = "path to input .cnf file")
parser.add_argument('-p', '--program', help = "program [ea.py, pso.py]", required = True)
parser.add_argument('-i', '--max_iterations',
                    nargs = '?', default = 1000, type = int,
                    help = "Maximal number of iterations. Default 1000")
parser.add_argument('-a','--list', nargs='+', help='Required algorithm to run [asap, rfea, flipga, sawea | psols, psosat, wpsosat]', required=True)
args = parser.parse_args()

#run algorithms
num = 0
files = []
for algorithm in args.list:
    print(algorithm, '...')
    files.append((algorithm, "{1}.txt".format(algorithm,num)))
    os.system('python3 {0} {1} {2} -i {3} > {4}.txt'.format(args.program, args.path, algorithm, args.max_iterations, num))
    num = num + 1
    print('done!')

color = iter(['-r','-g','-b','-y', '-k'])
plt.xlabel('Broj iteracija')
plt.ylabel('Broj zadovoljenih klauza')
#read data and add to plt
for algorithm, fileIn in files:
    with open(fileIn, 'r') as inputData:
        lines = inputData.readlines()
        lines = [line.replace('\n','') for line in lines]
        solution_str = filter(lambda elem: int(elem.isdigit()), lines)
        solution = [int(number) for number in solution_str]
        #print([*solution])
        c = next(color)
        plt.plot(range(1, len(solution)+1), solution, c, label = algorithm)
    os.system('rm {0}'.format(fileIn))

#plot
legend = plt.legend(loc='best', shadow=True, fontsize='x-large')
plt.show()
