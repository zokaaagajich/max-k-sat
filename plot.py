#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
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
    os.system('time python3 {0} {1} {2} -i {3} > {4}.txt'.format(args.program, args.path, algorithm, args.max_iterations, num))
    num = num + 1
    print('done!')


color = iter(['-r','-g','-b','-y', '-k'])
shape = iter(['o','x','v','^'])
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel('Broj iteracija', fontsize = 18)
plt.ylabel('Broj zadovoljenih klauza', fontsize = 18)

#read data and add to plt
for algorithm, fileIn in files:
    with open(fileIn, 'r') as inputData:
        lines = inputData.readlines()
        num_iteration = int(lines[-1].split()[1])

        solution = [*map(lambda digit_str: int(digit_str),
                     filter(lambda line: line.isdigit(),
                     map(lambda l: l.rstrip('\n'),
                        lines)))]
        #print(solution)

        c = next(color)
        if num_iteration == 1:
            c = c[1] + next(shape)
        plt.plot(range(1, num_iteration+1), solution, c, label = algorithm)

    os.system('rm {0}'.format(fileIn))


#plot
legend = plt.legend(loc='best', shadow=True, fontsize = 18)
plt.show()
