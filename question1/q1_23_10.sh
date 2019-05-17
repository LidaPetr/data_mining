#!/bin/bash
#$ -l h_rt=2:00:00  #time needed
#$ -pe smp 10 #number of cores
#$ -l rmem=15G #number of memery
#$ -o q1_23_10cores.output #This is where your output and errors are logged.
#$ -j n # normal and error outputs into a single file (the file above)
#$ -cwd # Run job from current directory
#$ -P rse-com6012
#$ -q rse-com6012.q

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 40g --executor-memory 2g --master local[10] --conf spark.driver.maxResultSize=4g ./Code/q1_23.py