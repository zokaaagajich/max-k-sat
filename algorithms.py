from random import randint
import os

# Functions returns array of clauses [[1,2,3], [-1,2], ... ]
# and number of literals(variables)
def extract_from_file(filename):
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
def binary_list(num, k):
    bin_str = bin(num)[2:].zfill(k)
    bin_list = [int(x) for x in bin_str]
    return bin_list

# Returns 1 if clause is true (satistied) or 0 if not satisfied.
def satisfied_clause(valuation_list, clause):
    values = [1-valuation_list[-clause[i]-1] if (clause[i]<0) else valuation_list[clause[i]-1]
                for i in range(0, len(clause))]

    return 0 if sum(values)==0 else 1

# BRUTE FORCE ALGORITHM
# For each combination from 0 to 2^num_of_vars count
# number of true clauses
def brute_force_algorithm(clauses, literals):
    n = 2**literals
    num_of_clauses = len(clauses)

    max = 0
    res_val_list = []

    for i in range(n):
        curr_max = 0
        valuation_list = binary_list(i, literals)
        for c in clauses:
            curr_max += satisfied_clause(valuation_list, c)

        if curr_max > max:
            max = curr_max
            res_val_list = valuation_list

        if max == num_of_clauses:
            break

    return (max, res_val_list)

# RANDOM ALGORITHM
# For given number of iterations gives random number
# for which we check number of true clauses.
# Does not garantee best solution.
def random_algorithm(clauses, literals, num_of_iters):
    n = 2**literals
    num_of_clauses = len(clauses)

    max = 0
    res_val_list = []

    for i in range(num_of_iters):
        random_num = randint(0,n)

        curr_max = 0
        valuation_list = binary_list(random_num, literals)

        for c in clauses:
            curr_max += satisfied_clause(valuation_list, c)

        if curr_max > max:
            max = curr_max
            res_val_list = valuation_list

        if max == num_of_clauses:
            break

    return (max, res_val_list)

def main():
    clauses, literals = extract_from_file(os.path.abspath("examples/input_easy.cnf"))
    max, val_list = brute_force_algorithm(clauses, literals)
    print(max, val_list)

    clauses, literals = extract_from_file(os.path.abspath("examples/input_sudoku.cnf"))
    max, val_list = random_algorithm(clauses, literals, 10)
    print(max, val_list)

if __name__ == "__main__":
    main()
