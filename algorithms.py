# Functions returns array of clauses [[1,2,3], [-1,2], ... ]
# and number of literals(variables)
def extractFromFile(filename):
    input = open(filename, "r")

    header = input.readline().split(" ")
    literals = int(header[2].rstrip())

    text = input.readlines()

    for i in range(0, len(text)):
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

# Check if satisfied clause
def satisfiedClause(valuation_list, clause):
    values = [1-valuation_list[-clause[i]-1] if (clause[i]<0) else valuation_list[clause[i]-1]
                for i in range(0, len(clause))]

    return 0 if sum(values)==0 else 1

def hardcore(clauses, literals):
    n = pow(2, literals)
    num_of_clauses = len(clauses)

    max = 0
    res_val_list = []

    # For each combination from 0 to 2^num_of_vars count how much clauses are true
    for i in range(0, n):
        curr_max = 0
        valuation_list = binary_list(i, literals)
        for c in clauses:
            curr_max += satisfiedClause(valuation_list, c)

        if curr_max > max:
            max = curr_max
            res_val_list = valuation_list

    return (max, res_val_list)


# main
clauses, literals = extractFromFile("input.cnf")
max, val_list = hardcore(clauses, literals)
print(max, val_list)
