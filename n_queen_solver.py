import numpy as np
import streamlit as st
from stqdm import stqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
######################################################################################
### Initi population #######
######################################################################################
def init_population(chromosome_size,population_size):
    population = []
    for _ in range(population_size):
     popul= np.random.permutation(range(1,chromosome_size+1))  # Random permutation of values from 1 to 8
     population.append(popul)
    return population
##########################################################################################
## The method for calculating the fitness of the population
###########################################################################################
def fitness(chrom,chromosome_size):
  q = 0
  for i1 in range(chromosome_size):
         tmp = i1 - chrom[i1]
         for i2 in range(i1+1,chromosome_size):
             q = q + (tmp == (i2 - chrom[i2]))
  for i1 in range(chromosome_size):
         tmp = i1 + chrom[i1]
         for i2 in range(i1+1,chromosome_size):
             q = q + (tmp == (i2 + chrom[i2]))
  return 1/(q+0.001)   
def mutation(chrom, chromosome_size):
    chrom_copy = chrom.copy()  # Create a copy of chrom
    ind1 = np.random.randint(0, chromosome_size)
    ind2 = np.random.randint(0, chromosome_size)
    if ind1 != ind2:
        tmp = chrom_copy[ind1]
        chrom_copy[ind1] = chrom_copy[ind2]
        chrom_copy[ind2] = tmp
    return chrom_copy
#######################################################################################
## Fitness curve 
#######################################################################################
def fitness_curve_plot(fitnees_curve):
   st.line_chart(fitnees_curve)
def fitness_curve_plot2(fitnees_curve):
   fig, ax = plt.subplots()
   n = np.linspace(0,len(fitnees_curve),len(fitnees_curve))
   plt.xlabel('Epoches')
   plt.ylabel('Fitness score')
   plt.plot(n, fitnees_curve,'-o')
   st.pyplot(fig)
##########################################################################################
## The method for plotting the result, the chess board and the queens
###########################################################################################
def n_queen_plot(chrom,chromosome_size):
    board_size = chromosome_size
    # Create a new figure and axis
    fig, ax = plt.subplots()
    # Set the aspect ratio of the plot to equal, to maintain square shape
    ax.set_aspect('equal')
    # Loop through each square on the chessboard
    for row in range(board_size):
        for col in range(board_size):
            # Calculate the coordinates of the square
            x = col
            y = row
            # Determine the color of the square based on its position
            color = 'white' if (row + col) % 2 == 0 else 'black'
            if chrom[row] == col+1 : color = 'red'
            # Create a rectangle patch for the square
            rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            # Add the rectangle to the plot
            ax.add_patch(rect)
    # Set the limits of the plot to display the entire chessboard
    ax.set_xlim(0, board_size)
    ax.set_ylim(0, board_size)
    # Remove the axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    # Show the plot
    #plt.show()
    st.pyplot(fig)
##########################################################################################
## The method for training the population
###########################################################################################
def train_population(population,epoches,chromosome_size):
    num_best_parents = 2
    ft = []
    success_booelan = False
    population_size = len(population)
    for i1 in stqdm(range(epoches),'Training.....'):   # 1 should be epoches later 
       fitness_score = []
       for i2 in range(population_size):
         fitness_score.append(fitness(population[i2],chromosome_size)) #fitness score initialisation
       ft.append(sum(fitness_score)/population_size)
       pop = np.concatenate((population, np.expand_dims(fitness_score, axis=1)), axis=1)
       sorted_indices = np.argsort(pop[:, -1])
       pop_sorted = pop[sorted_indices]
       pop = pop_sorted[:, :-1]
       best_parents_muted = []
       best_parents = pop[-num_best_parents:]
       best_parents_muted = [mutation(best_parents[i], chromosome_size) for i in range(num_best_parents)]
       pop[0:num_best_parents] = best_parents_muted
       population = pop
       if ft[-1] == 1000:  # this should be calculated accurately. In each case the model might pass the potimum solution, so whenever it is touching the solution's score we should stop the training
        print('Woowww, the model could find the solution!!')
        print('Here is an example of a solution : ',population[-1])
        success_booelan = True
        break
    return population, ft, success_booelan
##########################################################################################
## The main body block of the program
###########################################################################################
if __name__ == '__main__':
    suc_bool = True
    col1, col2, col3 = st.columns(3)
    
    with col1:
     st.title('The N-Queen Demonstration')
     chromosome_size = st.number_input('Insert chromosome size:')
     population_size = st.number_input('Insert population size:')
     epoches = st.number_input('Insert epoches:')
    population = init_population(int(chromosome_size),int(population_size))
    population, ft, suc_bool = train_population(population,int(epoches),int(chromosome_size))
    
    with col2:
     fitness_curve_plot2(ft)
     n_queen_plot(population[int(population_size)-1],int(chromosome_size))
    
    with col3:
      if suc_bool == False:
        st.text('Trained!')
        st.text('!!!!!!!!Opps, the model could not find the solution with te current epoches. Please whether increases them or increase the number of population_size.')   
      else:
        st.text('Trained!')
        st.text("Great!!!! We found a solution for you!")
        st.text("Here is an example:")
        st.text(population[-1]) 
