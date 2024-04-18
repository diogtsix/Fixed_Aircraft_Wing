# **Fixed_Aircraft_Wing**

## Overview 

This project focuses on Solving, Modelling and Simulating the Structural Dynamics and the Structural Optimization of a Fixed aircraft's Wing. It aims to the design of a full software architecture and the implementation of advanced engineering methods for real structures. All the project have been developed in Python with OOP. 

The project consist 2 distinct components and the interaction between those two : 

1) Structural Dynamics 
2) Structural Optimization

## **Keywords**

- Structural Dynamics
- Aircraft Wing
- NACA airfoil
- FEM
- Postprocessing
- Newmark
- Numerical Simulation
- Eigenanalysis
- Real Time Simulation

- Structural Optimization
- Genetic Algorithm 
- Particle Swarm Optimization
- Weight Minimization
- Neural Network 
- Machine Learning 
- Surrogate ML model Constrained Optimization

- GUI Design 
- Interactive GUI Design 
- Dynamic Implementation

- OOP
- Python


## **Structural Dynamics**

## Modelling and Solving

Using real dimensions for NACA airfoil, we model the structure using 3D bars and beams. Material, number of airfoils, wing length etc, are all parametrized for user input. In the next step we derive our **dynamic equations** using Finite Element Method (FEM). To solve these equations we have implement 2 different methds, we can solve either with Newmark method (numerical method for differential equations) or with eigenfrequency-eigenmode analysis (analytical solution). After calculating the displacements on each node, we calcualte strain and stresses for our structure. 

Regarding the loading of the structure we implemented harmonic loading in the form : F = F0 sin(Ωt) with Ω = 115.28 Hz. The magnitude of this harmonic excitation is parametrized and can be changed. 

## Visualization and Simulation

At the moment the software contains 3 distinct types of results visualization, using the GUI: 

1) Frequency response for specific node
2) Eigenmode visualization (for any number the user wants)
3) Real-Time displacement simulation 

Also inside the postprocessor there multiple other results to be visualized but not yet implemented inside the GUI. 

## Program Components

The basic components for a FEM program are : 1) Preprocessor, 2) Solver, 3) PostProcessor

For Graphical Visualization Purposes we Have also the Structural_Dynamics_main (GUI) where parameters are given as an input from the user. In the same GUI the real time simulation is visualizaed at the right of the panel. 

Also, fundamental structure components such as, nodes, elements etc, are implemented separately in the dataobjects folder in order to have a more organized OOP program. 

Mathematical Methods (Newmark, eigenAnalyis) are located inside the utilities folder. 

There is also a unitTests folder where we implement unit tests for proper component functioning. (At the moment, only for the preprocessor).

## **Structural Optimization**

## Modelling and Solving

Based on the Structural Dynamics Model we conduct element's material and surface optimization under constraints with the aim of Structure's weight minimization. We set discrete values and not continous for the optimization variables (sruface, materials).

Currently the constraints are based on the structure's stresses. For the optimization we have Implemented 2 different approaches : 

1) Objective Function constrained minimization
2) Surrogate Neural Network Model constrained minimization

We use Genetic Algorithm optimization for both the above models (also Particle Swarm Optimization will be completed soon), and we calcualte the best fit (optimal values for our vars) for our model. The implementation of GA have been conducted through DEAP library. 

For the Surrogate Model, we trained based on generated dataset a neural network, which in latter steps is used as an objective function for our model. 

## Program Components

The basic componenets for the optimzation are : 1) weight_optimization, 2) material_database, 3) optimization_GUI

All the calcualtions and functionalities are inside the weight_optimization file. The material_database is used to remove, replace or add more elements for the optimization. The .json file to add more materials is located inside the Utilities folder. 

Through the otpimization_GUI the user can set the parametrs for the Genetic algorithm, number of samples and discrete surface values for optimization. After optimization is finished report on the right of the same window will appear. 

## **Documentation**

An important componenet of this project is the documentation folder. Inside This folder there are 3 .pdf files: 

1) Simple_Beam_Structural_Dynamics_Analyis : Theoritical and Fundamental backgroud for Structural Dynamics implementation on a simple beem. Full analytical deriviatin for the equations. 

2) Fixed_Wing_Structural_Dynamics : The Structural Dynamics model which was developed by me in an older project. There you can find results and plots regarding the wing behaviour under specific loading. This report is considered as a benchmark for our model and it is the early steps I did around the implementation of wing's structural dynamics. 

3) Structural_Optimization_for_fixed_Wing : This document contains some theoritical background for the methods we have implemented. In this pdf you can also find our optimization project architecture and some code snippets. 

In general the above reports can guide you through the whole engineering methods and problem we faced and solved in this project. 


## **How To Run The Code**

The running process for this code is fairly simple. You just need to open the main.py file, which is located on the higher level for the project, and run it. This file will guide you through the steps. 

A window will appear firstly asking which problem you want to run (Structural Dunamics or Optimization), after this step the specific GUI will appear and you are ready to solve it. It is also possible after finishing let's say the structural Optimization, without closing th windows to conduct also a structural dunamics analysis (for real time simulation etc.), or after running on type of model optimization to run also the second one and the report will display both results for comparison

## **Libraries Used**

1) Sci-Kit Learn
2) Pandas
3) Tensorflow
4) Keras
5) DEAP (for genetic algorithm)
6) Numpy 
7) matplotlib
8) PyQt5 (for GUI design)
9) scipy (for the FEM model)

8) pyswarm (For Particle Swarm Optimization, but not yet functional)
