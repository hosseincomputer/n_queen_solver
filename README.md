# N-Queen Problem Solver

This Python script solves the N-Queen problem using a Genetic Algorithm (GA) approach. It aims to find a placement of N queens on an NxN chessboard such that no two queens threaten each other.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:
- numpy
- matplotlib
- argparse
- tqdm


You can install these dependencies by running the following command:

```
pip install -r requirements.txt
```

## Usage

To run the script, use the following command:

```
python n_queen_solver.py <chromosome_size> <population_size> <epoches>
```

- `<chromosome_size>`: The size of a chromosome (the number of queens).
- `<population_size>`: The size of the population of chromosomes.
- `<epoches>`: The number of iterations to train the GA model.

HINT: There is no soluiton for N=1,2,3. Please consider a number larger or equal than 4 for testing. 

## Functionality

The script implements the following functions:

- `fitness(chrom, chromosome_size)`: Calculates the fitness score of a chromosome based on the number of non-attacking pairs of queens.
- `mutation(chrom, chromosome_size)`: Performs mutation on a chromosome by swapping two random positions.
- `fitness_curve_plot(fitness_curve)`: Plots the fitness curve during the training process.
- `n_queen_plot(chrom, chromosome_size)`: Displays the chessboard and the placement of queens for a given chromosome.
- `init_population(chromosome_size, population_size)`: Initializes the population of chromosomes randomly.
- `train_population(population, epoches, chromosome_size)`: Trains the population using the GA model for a specified number of epochs.

## Output

The script will output the following:

- The fitness curve plot, showing the improvement of the fitness score over the training iterations.
- The final placement of queens on the chessboard for the best solution found.
The folder images contains two subfolders of learning_curve and solutions with recorded several tests of the script. 

Enjoy solving the N-Queen problem using the Genetic Algorithm!

For more details on the implementation and techniques used, please refer to the accompanying documentation and articles.
