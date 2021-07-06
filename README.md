# brain_connectivity
Optimal transport for comparing short brain connectivity between individuals
first name: Duy Anh Philippe
last name: Pham
email: duyanhphilippe.pham[at]gmail.com

python 3.7
version 1
start: February 8, 2021
end: July 31, 2021

# Presentation of the projet
The first objective of our work is to build the best group subject. To achieve that, we will use tools from the optimal transport that will allow us to have a better alignment between individuals
and to generate a better quality group profile. The choice to work with the theory of optimal transport is motivated by the fact that the different connectivity maps can be seen as different
probability distributions. The goal of optimal transport is to define the least costly transformation from one distribution to another. 
This allows us to determine the group profile as the barycenter of all the individual profiles, in the sense of optimal transport, and thus to project
them onto the group profile.
Our secondary objective is to study stratification within the population to see if it exists. For this we used the kmedoids and the isomap.

You can find the report in English [here](https://github.com/Daphilippe/brain_connectivity/blob/main/Presentation/english%20report.pdf) and the presentation in English [here](https://github.com/Daphilippe/brain_connectivity/blob/main/Presentation/english%20presentation.pdf).
You can find the report in French [here](https://github.com/Daphilippe/brain_connectivity/blob/main/Presentation/rapport%20français.pdf) and the presentation in French [here](https://github.com/Daphilippe/brain_connectivity/blob/main/Presentation/présentation%20français.pdf).

# Organization of the project
## Data
Data from 100 subjects from each hemisphere generated by Alexandre Pron (https://github.com/alexpron)
## libs
Internal project library
## variables
Intermediate data generated during the project. 
This corresponds to different experiments with conservation of intermediate results
## Presentation
Presentation of the work done as part of an end-of-study project as part of a double degree 
between [CPE Lyon](](https://www.cpe.fr/en/)) and [Jean Monnet University](https://mldm.univ-st-etienne.fr) to the [MeCa team](https://meca-brain.org).

# Libraries used
* matplotlib 
* operator
* numpy
* os
* scipy
* pot (https://pythonot.github.io version 0.7.0)