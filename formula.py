import math

variable_valuation = {}

class Formula:
    def __init__(self):
        pass

class T(Formula):
    def __str__(self):
        return 'T'

    def interpretation(self):
        return True

class F(Formula):
    def __str__(self):
        return 'F'

    def interpretation(self):
        return False

class Letter(Formula):
    def __init__(self, letter):
        self.letter = letter
        variable_valuation[self.letter] = True

    def __str__(self):
        return self.letter

    def interpretation(self):
        return variable_valuation[self.letter]

class Not(Formula):
    def __init__(self, op1):
        self.op1 = op1

    def __str__(self):
        return "~(%s)" % self.op1.__str__()

    def interpretation(self):
        return not self.op1.interpretation()

class And(Formula):
    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2

    def __str__(self):
        return "(%s & %s)" % (self.op1.__str__(), self.op2.__str__())

    def interpretation(self):
        return self.op1.interpretation() and self.op2.interpretation()

class Or(Formula):
    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2

    def __str__(self):
        return "(%s | %s)" % (self.op1.__str__(), self.op2.__str__())

    def interpretation(self):
        return self.op1.interpretation() or self.op2.interpretation()



F00 = T()
F01 = F()
F02 = Letter('p')
F03 = Not(F00)
F04 = And(F00, F01)
F05 = Or(Not(F02), F01)

print(F00, F00.interpretation())
print(F01, F01.interpretation())
print(F02, F02.interpretation())
print(F03, F03.interpretation())
print(F04, F04.interpretation())
print(F05, F05.interpretation())
