Software requirements
----------------------

- python2.7
- numpy >= 1.8.0
- scipy >= 0.13.2
- statsmodels >= 0.8.0
- rpy2 >= 2.8.5
- pickle 

Instructions to run the simulator
=================================

Grid configuration
-------------------

For each site in the grid, specify the location, name, number of cores, power per core (in watts), time zone, GFLOPS, maximum value of estimated run time, and maximum job size in the file run_simulator.py. For example, in the XSEDE grid, the parameters for Blacklight are:

        Blacklight.location = "Pittsburgh"
        Blacklight.name = "Blacklight"
        Blacklight.cores = 4096
        Blacklight.powerPerCore = 87.89 # Watts
        Blacklight.timeZoneLag = 0
        Blacklight.gflops = 9.0
        Blacklight.max_ert = 86400
        Blacklight.max_req = 1024

The electricty price data for the specified locations should be present in the directory priceSuite/ 
For more information see XSEDE grid configuration in run_simulator.py.

Sample execution command
------------------------

python2.7 run_simulator.py --input-file=test.swf --scheduler=4 --maxq=5 --mperc=100 --joblimit=10 --alpha=2500 --beta=7500 --algorithm=2 --pfile=test.pkl -pricepath=./priceSuite/price_data/XSEDE/

input-file - Supply a grid workload in Standard Workload Format (SWF). Each job line in the SWF format is appened by the site id of the location where the job originated. 

scheduler - 4 denotes EASY scheduler

maxq - Max. limit on the number of metascheduling jobs routed to a particular site in any given scheduling cycle

mperc - Percentage of jobs which are grid submissions

joblimit - Number of jobs processed (excludes history size)

alpha, beta - MCMF algorithm parameters (Supply desired integer values to these parameters; alpha/(alpha+beta) is taken as the weight for response time and beta/(alpha+beta) is the weight for the electricty price. In the example, 25% is the weight for response time)

algorithm = 1 (MCMF), 2 (TWOPRICE), 3 (INST)

pfile = Python Pickle file containing job statistics

pricepath = directory containing the electricity price data for the locations specified in run_simulator.py.


Instructions to run the predictors
==================================

Instructions to run the individual predictors for queue waiting times and electricity prices are contained in the README files located in the corresponding directories under the predictors/ directory.

Primary Author
==============
Prakash Murali. Developed when he was a Masters student in MARS lab, SERC, IISc. Continuing development as he is currently working in IBM research, Bangalore.

Sanity Checking
===============
Sathish Vadhiyar, Associate Professor, CDS and SERC, IISc. Convenor of MARS lab, CDS (earlier SERC), IISc.

Expansions
==========
CDS - Department of Computational and Data Sciences
SERC - Supercomputer Education Research Centre
MARS - Middleware and Runtime Systems Lab
IISc - Indian Institute of Science, Bangalore
# mcmf-metascheduler-predictors
A repository for metascheduling of HPC jobs in computational grids
