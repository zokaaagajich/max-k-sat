#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import argparse
import os

from timeit import default_timer as timer

#Parsing arguments of command line
parser = argparse.ArgumentParser()
parser.add_argument('path', help = "path to input .cnf file")
parser.add_argument('-s','--program', nargs='+', help='list of programs to start: ea, pso', required=True)
parser.add_argument('-i', '--max_iterations',
                    nargs = '?', default = 1000, type = int,
                    help = "Maximal number of iterations. Default 1000")
parser.add_argument('-e','--listEA',  nargs='+', help='Algorithms to run for EA: [asap, rfea, flipga, sawea', required=False)
parser.add_argument('-p','--listPSO', nargs='+', help='Algorithm to run for PSO: [psols, psosat, wpsosat]', required=False)
args = parser.parse_args()

#fout = open('results.csv', 'a+')
time_list = []

#run algorithms
num = 0
files = []
for program in args.program:
    if program[0] == 'e':
        for algorithm in args.listEA:
            print(algorithm, '...')
            files.append((algorithm, "{1}.txt".format(algorithm,num)))
            start = timer()
            os.system('time python3 {0}.py {1} {2} -i {3} > {4}.txt'.format(program, args.path, algorithm, args.max_iterations, num))
            end = timer()
            time_list.append(end-start)
            num = num + 1
            print('done!')

    if program[0] == 'p':
        for algorithm in args.listPSO:
            print(algorithm, '...')
            files.append((algorithm, "{1}.txt".format(algorithm,num)))
            start = timer()
            os.system('time python3 {0}.py {1} {2} -i {3} > {4}.txt'.format(program, args.path, algorithm, args.max_iterations, num))
            end = timer()
            time_list.append(end-start)
            num = num + 1
            print('done!')

t = iter(time_list)

color = iter(['-r','-g','-b','-y', '-k', '-c', '-m', '-w'])
shape = iter(['o','x','v','^', '<', '>', '1', '2', '3'])
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

        #fout.write(str(sum(solution)/num_iteration))
        #fout.write(',')
        #fout.write(str(next(t)/num_iteration))
        #out.write(',')

        c = next(color)
        if num_iteration == 1:
            c = c[1] + next(shape)
        plt.plot(range(1, num_iteration+1), solution, c, label = algorithm)

    os.system('rm {0}'.format(fileIn))

#fout.write('\n')
#fout.close()

#plot
legend = plt.legend(loc='best', shadow=True, fontsize = 18)
plt.show()
