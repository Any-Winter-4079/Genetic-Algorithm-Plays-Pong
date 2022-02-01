#################
#####  GA   #####
#################

import numpy as np
from math import ceil
from copy import deepcopy
from abc import ABC, abstractmethod
import sys


class GA(object):
    """Genetic Algorithm.
       -> Create a GA object defining: function to minimize, acceptable min, max values for each variable (list of tuples) and population size.
       -> Run calling the run() method of the Object created defining the iterations to run for.
       -> myGA = GA(sphere, [(0,10),(2,20)]), 50)
       -> myGA.run(500)
    """

    def __init__(self, minfun, bounds, psize):
        """Construct Genetic Algorithm (GA) Object"""
        super(GA, self).__init__()
        self.minfun = minfun  # function to minimize
        # acceptable bounds (min, max) for each variable: e.g. [(0,3), (1,10)] -> 2 variables x:[0,3], y:[1,10]
        self.bounds = bounds
        self.psize = psize  # population size

    @classmethod
    def create_test_population(cls, dtype="continuous"):
        psize = np.random.randint(50, 100)
        n_genes = np.random.randint(5, 40)
        bounds = [(np.random.randint(0, 15), np.random.randint(15, 25))
                  for i in range(n_genes)]  # integer bounds
        p = Population(bounds, psize, dtype=dtype)
        print(" -> Population size: {}".format(psize))
        print(" -> # of genes per genome: {}".format(n_genes))
        print(" -> Genes bounds: {}".format(bounds))
        sys.stdout.flush()
        return psize, n_genes, bounds, p

    @classmethod
    def test_population(cls, dtype="continuous"):
        """Test creation of a Population."""
        print("Testing Population creation:")
        psize, n_genes, bounds, p = GA.create_test_population(dtype=dtype)
        for genome in p.genomes:
            print("  -> Genome: {}".format(genome.__dict__))
            assert (genome.fitness)
            print("   Ok. Genome passes fitness test.")
            assert (len(genome.genes) == n_genes)
            print("   Ok: Genome passes no. of genes test.")
            for idx in range(len(genome.genes)):
                print("   Gene: {}".format(genome.genes[idx]))
                print("   Bounds: {}".format(bounds[idx]))
                assert (genome.genes[idx] >= bounds[idx][0])
                assert (genome.genes[idx] <= bounds[idx][1])
                print("   Ok: Gene passes test.")
            print("  Ok: Genome passes test.")
        print("Ok: All tests passed.")
        sys.stdout.flush()

    @classmethod
    def test_two_point_operator(cls):
        """Test Two Point Operator (a type of Crossover Operator)."""
        print("Testing Two Point Operator:")
        psize, n_genes, bounds, p = GA.create_test_population()
        n_parents = 2
        couples = [p.genomes[i:i + n_parents] for i in range(0, psize, n_parents)][
            :-1]  # [:-1] to ensure no single genome at last index
        for couple in couples:
            TwoPoint.apply(couple, debug=True)
        print("Ok: All tests passed.")
        sys.stdout.flush()

    @classmethod
    def test_uniform_operator(cls, dtype="continuous"):
        """Test Uniform Operator (a type of Mutation Operator)."""
        print("Testing Uniform Operator:")
        psize, n_genes, bounds, p = GA.create_test_population(dtype=dtype)
        for genome in p.genomes:
            print(" -> Genome: {}".format(genome.__dict__))
            mutation_chance = np.random.uniform(0, 0.25)
            print(" -> Mutation chance: {}".format(mutation_chance))
            Uniform.apply([genome], mutation_chance=mutation_chance,
                          dtype=dtype, debug=True)
            print("Ok: All tests passed.")
        sys.stdout.flush()

    @classmethod
    def test_rulette_wheel_operator(cls):
        """Test Rulette Wheel Operator (a type of Selection Operator)."""
        print("Testing Rulette Wheel Operator:")
        print("Test 1:")
        psize, n_genes, bounds, p = GA.create_test_population()
        for genome in p.genomes:
            genome.fitness = 0
        RouletteWheel.apply(p, debug=True)  # test with 0 fitness
        print("Ok: Test 1 passed.")
        print("Test 2:")
        for genome in p.genomes:
            genome.fitness = np.random.uniform(-100, 100)
        RouletteWheel.apply(p, debug=True)  # test with different fitness
        print("Ok: Test 2 passed.")
        print("Test 3:")
        choice = int(np.random.choice(len(p.genomes), 1))
        print(" -> Chosen index: {}".format(choice))
        p.genomes[choice].fitness = np.random.uniform(-20000, -15000)
        # visual test with fitness with a high chance of being selected
        RouletteWheel.apply(p, debug=True)
        print("Ok: All tests passed.")
        sys.stdout.flush()

    @classmethod
    def test_elitist_operator(cls):
        """Test Elitist Operator (a type of Replacement Operator)."""
        print("Testing Elitist Operator:")
        print("Test 1:")
        psize, n_genes, bounds, p1 = GA.create_test_population()
        for genome in p1.genomes:
            genome.fitness = np.random.uniform(-100, 100)
        Elitist.apply(p1, p1, debug=True)  # test with same population
        print("Ok: Test 1 passed.")
        print("Test 2:")
        p2 = deepcopy(p1)
        for genome in p2.genomes:
            genome.fitness = np.random.uniform(-100, 100)
        Elitist.apply(p1, p2, debug=True)  # test with different population
        print("Ok: Test 2 passed.")
        print("Ok: All tests passed.")
        sys.stdout.flush()

    def run(self, iters, dtype="continuous", debug=False):
        p = Population(self.bounds, self.psize,
                       dtype=dtype)  # parent population
        #self.set_fitness(p)
        for _ in range(iters):
            print("Generation: {}".format(_ + 1))
            sys.stdout.flush()
            if debug:
                for genome in p.genomes:
                    print(genome.__dict__)
                sys.stdout.flush()
            p2 = deepcopy(p)
            for i in range(len(p.genomes) // 2 + 1):
                if debug:
                    print("Reproduction: {}".format(i + 1))
                parent1 = RouletteWheel.apply(p)[0]  # select genome
                if debug:
                    print("Parent1: {}".format(parent1.__dict__))
                parent2 = RouletteWheel.apply(p)[0]  # select genome
                if debug:
                    print("Parent2: {}".format(parent2.__dict__))
                    sys.stdout.flush()
                child1, child2 = TwoPoint.apply(
                    [parent1, parent2])  # reproduce genomes
                if debug:
                    print("Child1: {}".format(child1.__dict__))
                    print("Child2: {}".format(child2.__dict__))
                    sys.stdout.flush()
                Uniform.apply([child1], dtype=dtype)  # mutate genome
                Uniform.apply([child2], dtype=dtype)  # mutate genome
                if debug:
                    print("Mutated Child1: {}".format(child1.__dict__))
                    print("Mutated Child2: {}".format(child2.__dict__))
                    sys.stdout.flush()
                p2.genomes[i] = child1  # add to children population
                p2.genomes[-i] = child2  # add to children population
            # Obtain fitness for both populations, because on some problems, same genes
            # may give different fitness [e.g. game]. This way, you don't continue including
            # lucky individuals without any reassessment (reevaluating fitness again to double check).
            # Additionally, we could evaluate fitness function several times, and take average,
            print("Parents")
            sys.stdout.flush()
            self.set_fitness(p)
            print("Children")
            sys.stdout.flush()
            self.set_fitness(p2)
            if debug:
                print("p Genomes:")
                for genome in p.genomes:
                    print(genome.__dict__)
                sys.stdout.flush()
                print("p2 Genomes:")
                for genome in p2.genomes:
                    print(genome.__dict__)
                sys.stdout.flush()
            # replace parent population with child population
            p = Elitist.apply(p, p2)
            self.bestGenome = p.genomes[0].genes
            if debug:
                print("Selected:")
                for genome in p.genomes:
                    print(genome.__dict__)
                sys.stdout.flush()
            print("######")
            print("Best: {}".format(
                min([genome.fitness for genome in p.genomes])))
            print("######")
            print("Mean: {}".format(
                np.mean(np.array([genome.fitness for genome in p.genomes]))))
            print("######")
            sys.stdout.flush()

    def set_fitness(self, p):
        """Sets the fitness for all genomes in the population"""
        for genome in p.genomes:
            genome.fitness = self.minfun(genome.genes)

    def best(self):
        """Best genome of final population. About this:
           -> Calculating the average fitness (with several training games)
              and testing individuals' fitness after every generation
              may reduce luck-driven noise enough to return best overall
              genome, though
        """
        return self.bestGenome


class Population():
    """Creates a new Population of Genomes.
       -> Genome 1: Fitness: +∞, Genes: [0, 3, ..., 7] as Genome 1 fitness, [var1, var2, ..., varn], for an n-variable problem.
       -> ...
       -> Genome n: Fitness: +∞, Genes: [2, 5, ..., 10] as Genome n fitness, [var1, var2, ..., varn], for an n-variable problem.
       -> Holds a list of Genomes.
    """

    def __init__(self, bounds, psize, dtype="continuous"):
        """Constructor that initializes a population."""
        self.genomes = [Genome(bounds, dtype=dtype) for i in range(psize)]


class Genome():
    """Creates a new Genome in the Population. Each genome is a candidate solution.
       -> Fitness: +∞, Genes: [0, 3, ..., 7]], as Genome fitness, [var1, var2, ..., varn], for an n-variable problem.
       -> Each gene (variable) is initialized randomly within its acceptable bounds.
       -> True candidate fitness is obtained only after testing the genome on the function.
    """

    def __init__(self, bounds, dtype="continuous"):
        """Constructor that initializes an individual."""
        n_genes = len(bounds)
        if dtype == "continuous":
            self.genes = np.random.uniform(low=[i[0] for i in bounds], high=[i[1] for i in bounds],
                                           size=n_genes)  # float valued genes (+0 on high b/c of continuous bound)
        else:
            self.genes = np.random.randint(low=[i[0] for i in bounds], high=[i[1] + 1 for i in bounds],
                                           size=n_genes)  # integer valued genes (+1 on high to include high bound)
        self.fitness = np.inf
        self.bounds = bounds  # sense of bounds to mutate within


class CrossoverOperator(ABC):
    """Defines an interface for a Crossover Operator.
       -> Abstract class.
       -> Implementations take a collection of genomes and return another collection of (as many, more, or less) genomes.
       -> Number of parents per reproduction and number of children created is implementation-dependant.
       -> Parents reproduce (recombine their genes, or variable values) and produce offspring.
    """

    @classmethod
    @abstractmethod
    def apply(cls, genomes, debug=False):
        pass


class TwoPoint(CrossoverOperator):
    """Returns a collection of 2 offspring (genomes) from a collection of 2 parents (genomes).
       -> Implements CrossoverOperator.
       -> A parent is a collection of genes (variables) and its fitness value:
          Parent 1: Fitness: 46, Genes: [0, 3, 10, 4, 8, ..., 7] as Genome fitness, [var1, var2, var3, var4, ..., varn]
          Parent 2: Fitness: 64, Genes: [2, 6, 1, 5, 9, ..., 10] as Genome fitness, [var1, var2, var3, var4, ..., varn]
       -> Offspring genes alternate parents' genes from 2 randomly selected cutting points
       -> Cutting points: index 2, index 4
       -> Parent genes:
          Parent 1: [0, 3] [10, 4] [8, ..., 7]
          Parent 2: [2, 6]  [1, 5] [9, ..., 10]
      -> Offspring genes:
          Offspring 1: [0, 3] [1, 5]  [8, ..., 7]  -> Fitness: +∞, Genes: [0, 3, 1, 5, 8, ..., 7]
          Offspring 2: [2, 6] [10, 4] [9, ..., 10] -> Fitness: +∞, Genes: [2, 6, 10, 4, 9, ..., 10]
      -> True candidate fitness is obtained only after testing the genome on the function.
    """

    @classmethod
    def apply(cls, genomes, debug=False):
        n_genes = len(genomes[0].genes)
        if n_genes < 2:
            return genomes
        n_cutting_points = 2
        cutting_points = np.sort(np.random.default_rng().choice(n_genes, size=n_cutting_points,
                                                                replace=False))  # ordered and unique cutting points
        parent1_chunks = [genomes[0].genes[0:cutting_points[0]],
                          genomes[0].genes[cutting_points[0]                                           :cutting_points[1]],
                          genomes[0].genes[cutting_points[1]:]
                          ]
        parent2_chunks = [genomes[1].genes[0:cutting_points[0]],
                          genomes[1].genes[cutting_points[0]                                           :cutting_points[1]],
                          genomes[1].genes[cutting_points[1]:]
                          ]
        child1 = deepcopy(genomes[0])
        child2 = deepcopy(genomes[1])
        child1.genes = np.concatenate(
            [np.array(parent1_chunks[0]), np.array(parent2_chunks[1]), np.array(parent1_chunks[2])])
        child2.genes = np.concatenate(
            [np.array(parent2_chunks[0]), np.array(parent1_chunks[1]), np.array(parent2_chunks[2])])
        child1.fitness = np.inf
        child2.fitness = np.inf
        if debug:
            print(" -> Cutting points: {}".format(cutting_points))
            print(" -> Parent1 genes: {}".format(genomes[0].genes))
            print(" -> Parent2 genes: {}".format(genomes[1].genes))
            print(" -> Parent1 chunks: {}".format(parent1_chunks))
            print(" -> Parent2 chunks: {}".format(parent2_chunks))
            print(" -> Child1 genes: {}".format(child1.genes))
            print(" -> Child2 genes: {}".format(child2.genes))
            assert (len(child1.genes) == len(
                genomes[0].genes) == len(genomes[1].genes))
            print(" Ok: Child 1 passes length test.")
            assert (len(child2.genes) == len(
                genomes[0].genes) == len(genomes[1].genes))
            print(" Ok: Child 2 passes length test.")
            assert ((child1.genes[0:cutting_points[0]] ==
                     genomes[0].genes[0:cutting_points[0]]).all())
            assert ((child1.genes[cutting_points[0]:cutting_points[1]] == genomes[1].genes[
                cutting_points[0]:cutting_points[1]]).all())
            assert ((child1.genes[cutting_points[1]:] ==
                     genomes[0].genes[cutting_points[1]:]).all())
            print(" Ok: Child 1 passes parent inheritance test.")
            assert ((child2.genes[0:cutting_points[0]] ==
                     genomes[1].genes[0:cutting_points[0]]).all())
            assert ((child2.genes[cutting_points[0]:cutting_points[1]] == genomes[0].genes[
                cutting_points[0]:cutting_points[1]]).all())
            assert ((child2.genes[cutting_points[1]:] ==
                     genomes[1].genes[cutting_points[1]:]).all())
            print(" Ok: Child 2 passes parent inheritance test.")
        sys.stdout.flush()
        return [child1, child2]


class MutationOperator(ABC):
    """Defines an interface for a Mutation Operator.
       -> Abstract class.
       -> Implementations take a collection of (one or more) genomes and return a genome with each gene potentially mutated.
    """

    @classmethod
    @abstractmethod
    def apply(cls, genomes, mutation_chance=0.2, dtype="continuous", debug=False):
        pass


class Uniform(MutationOperator):
    """Returns a genome with each gene potentially mutated based on some mutation chance.
       -> Implements MutationOperator.
       -> Takes a collection of one single genome:
          [Fitness: 46, Genes: [0, 3, ..., 7]  as Genome fitness, [var1, var2, ..., varn]]
       -> Mutates each gene with a certain probability:
           Fitness: +∞, Genes: [0, 5, ..., 7]  as Genome fitness, [var1, var2, ..., varn]
       -> True candidate fitness is obtained only after testing the new genome on the function.
    """

    @classmethod
    def apply(cls, genomes, mutation_chance=0.2, dtype="continuous", debug=False):
        genome = genomes[0]
        n_genes = len(genome.genes)
        mutation_on_index = np.random.choice([True, False], n_genes, p=[mutation_chance,
                                                                        1 - mutation_chance])  # [False, False, True, ... False]
        if debug:
            print(" -> Mutation on gene index: {}".format(mutation_on_index))
        for i in range(n_genes):
            if mutation_on_index[i]:
                if dtype == "continuous":
                    new_gene = np.random.uniform(low=genome.bounds[i][0], high=genome.bounds[i][1],
                                                 size=(1,))  # float valued genes (+0 on high b/c of continuous bound)
                else:
                    new_gene = np.random.randint(low=genome.bounds[i][0], high=genome.bounds[i][1] + 1,
                                                 size=1)  # integer valued genes (+1 on high to include high bound)
                old_genes = deepcopy(genome.genes)
                genome.genes[i] = new_gene
                if debug:
                    print(" -> Mutation index: {}".format(i))
                    print(" -> New gene: {}".format(new_gene))
                    assert (new_gene >= genome.bounds[i][0])
                    assert (new_gene <= genome.bounds[i][1])
                    print(" -> Old genes: {}".format(old_genes))
                    print(" -> New genes: {}".format(genome.genes))
                    for idx in range(n_genes):
                        if i != idx:
                            assert (genome.genes[idx] == old_genes[idx])
                            print(" Ok: Gene passes test.")
                    print(" Ok: Genome passes test.")
        sys.stdout.flush()
        return genome


class SelectionOperator(ABC):
    """Defines an interface for a Selection Operator.
       -> Abstract class.
       -> Implementations take a population of genomes and an optional target vector position, and return a collection of genomes.
    """

    @classmethod
    @abstractmethod
    def apply(cls, p, i=None, debug=False):
        pass


class RouletteWheel(SelectionOperator):
    """Returns a collection of a single genome chosen probabilistically out of a population of genomes, based on fitness.
       -> Implements SelectionOperator.
       -> Takes a population of genomes:
          Fitness: 46, Genes: [0, 3, ..., 7]  as Genome fitness, [var1, var2, ..., varn]
          Fitness: 64, Genes: [2, 5, ..., 10] as Genome fitness, [var1, var2, ..., varn]
          Fitness: 14, Genes: [4, 7, ..., 12] as Genome fitness, [var1, var2, ..., varn]
       -> Chooses a genome probabilistically:
          46 -> (1/46) / (1/64 + 1/46 + 1/14) -> Probability of selection: 0.199...
          64 -> (1/64) / (1/64 + 1/46 + 1/14) -> Probability of selection: 0.143...
          14 -> (1/14) / (1/64 + 1/46 + 1/14) -> Probability of selection: 0.656...
       -> Accounting for 0 / negative fitness:
          Subtract Minimum Fitness   Fitness = Max - Fitness
           8 ->  8 - (-2) -> 10 ->    (10 - 10) -> 0      ->   0  / (0+4+10+8) -> Probability of selection: 0 (discard)
           4 ->  4 - (-2) -> 6  ->    (10 - 6)  -> 4      ->   4  / (0+4+10+8) -> Probability of selection: 0.181...
          -2 -> -2 - (-2) -> 0  ->    (10 - 0)  -> 10     ->   10 / (0+4+10+8) -> Probability of selection: 0.454...
           0 ->  0 - (-2) -> 2  ->    (10 - 2)  -> 8      ->   8  / (0+4+10+8) -> Probability of selection: 0.363...
       -> Where less "fitness" is better (minimization problem).
    """

    @classmethod
    def apply(cls, p, i=None, debug=False):
        if len(p.genomes) < 2:  # we need minimum 2 genomes
            return [p.genomes[0]]  # assuming non empty
        if all(p.genomes[i].fitness == p.genomes[-i].fitness for i in
               range(len(p.genomes))):  # we need minimum 2 diff fitness values to operate
            choice = int(np.random.choice(len(p.genomes), 1))  # random index
            if debug:
                print(" -> No. of genomes: {}.".format(len(p.genomes)))
                print(" -> Choice index: {}".format(choice))
                assert (choice >= 0)
                assert (choice < len(p.genomes))
                print(" -> Ok: Choice index passes test")
            return [p.genomes[choice]]
        minimum = min([genome.fitness for genome in p.genomes])
        # positive scale [0, +∞)
        normalized_fitness = [genome.fitness - minimum for genome in p.genomes]
        maximum = max(normalized_fitness)
        # fitness for min problem (inverted)
        min_fitness = [maximum - fitness for fitness in normalized_fitness]
        probs = [fitness / sum(min_fitness) for fitness in min_fitness]
        # chosen genome index
        choice = int(np.random.choice(len(p.genomes), 1, p=probs))
        if debug:
            print(" -> Minimum Fitness: {}".format(minimum))
            for genome in p.genomes:
                print(" -> Genome Fitness: {}".format(genome.fitness))
                assert (genome.fitness >= minimum)
            print(" -> Ok: Minimum fitness passes test.")
            print(" -> Normalized Fitness: {}".format(normalized_fitness))
            print(" -> Maximum Normalized Fitness: {}".format(maximum))
            for fitness in normalized_fitness:
                assert (fitness >= 0)
                assert (fitness <= maximum)
            print(" -> Ok: Normalized fitness passes Zero test")
            print(" -> Ok: Normalized fitness passes Maximum test.")
            print(" -> Fitness for Minimization problem: {}".format(min_fitness))
            print(" -> Probabilities: {}".format(probs))
            print(" -> Sum of probs: {}".format(sum(probs)))
            assert (sum(probs) + 0.01 >= 1)
            assert (sum(probs) - 0.01 <= 1)
            print(" -> Ok: Probs pass add up to 1 test.")
            print(" -> Choice index: {}".format(choice))
            assert (choice >= 0)
            assert (choice < len(p.genomes))
            print(" -> Ok: Choice index passes test")
        sys.stdout.flush()
        return [p.genomes[choice]]


class ReplacementOperator():
    """Defines an interface for a Replacement Operator.
       -> Abstract class.
       -> Implementations take two populations of genomes and return a single population of genomes.
    """

    @classmethod
    @abstractmethod
    def apply(cls, p1, p2, debug=False):
        pass


class Elitist(ReplacementOperator):
    """ Returns a new population made up of the best genomes out of both taken populations.
       -> Implements SelectionOperator.
       -> Takes 2 populations:
          Population 1: Fitness: 46, Genes: [...], Fitness: 64, Genes: [...], Fitness: 20, Genes: [...]
          Population 2: Fitness: 36, Genes: [...], Fitness: 54, Genes: [...], Fitness: 50, Genes: [...]
       -> Returns 1 population:
          Population:   Fitness: 20, Genes: [...], Fitness: 36, Genes: [...], Fitness: 46, Genes: [...]
       -> Size of population is maintained.
    """

    @classmethod
    def apply(cls, p1, p2, debug=False):
        p_cct = p1.genomes + p2.genomes  # concatenate
        if debug:
            print(" -> Concatenated population size: {}.".format(len(p_cct)))
            assert (len(p1.genomes) + len(p2.genomes) == len(p_cct))
            print(" -> Concatenated population passes size test.")
        p_cct_sort = sorted(p_cct, key=lambda genome: genome.fitness)
        if debug:
            for i in range(len(p_cct_sort) - 1):
                print(
                    " -> Curr genome fitness: {}.".format(p_cct_sort[i].fitness))
                print(
                    " -> Next genome fitness: {}.".format(p_cct_sort[i + 1].fitness))
                assert (p_cct_sort[i].fitness <= p_cct_sort[i + 1].fitness)
                print(" -> Genomes pass sorted test.")
            print(" -> Concatenated population passes sorted test.")
        p = deepcopy(p1)
        p.genomes = p_cct_sort[:(len(p1.genomes))]
        if debug:
            print(" -> Population size: {}.".format(len(p.genomes)))
            assert (len(p.genomes) == len(p1.genomes) == len(p2.genomes))
            print(" -> Population size passes test.")
        sys.stdout.flush()
        return p

#################
###  TESTING  ###
#################
# GA.test_population(dtype="discrete")
# GA.test_population(dtype="continuous")
# GA.test_two_point_operator()
# GA.test_uniform_operator(dtype="discrete")
# GA.test_uniform_operator(dtype="continuous")
# GA.test_rulette_wheel_operator()
# GA.test_elitist_operator()

#################
###  RUNNING  ###
#################
# Some functions for testing:
# -> sphere
# -> ackley
# -> rosenbrock
# -> rastrigin
# -> griewank
# -> schwefel_2_21
# -> schwefel_2_22
# -> schwefel_1_2
# -> extended_f_10
# -> bohachevsky
# -> schaffer

# Or build a custom function:
# def sum_fun(v):
#     return sum(v)

# mybounds=[(0,10)]*5
# myGA=GA(sum_fun, mybounds, 6)
# myGA.run(20, dtype="discrete")
# myGA.run(100, dtype="continuous")
# myGA.run(100)
# bestGenome=myGA.best()
# print(bestGenome)
