#!/bin/bash
#$ -N SurfaceXSyndrome
#$ -cwd
## Set this to number of jobs (number of samples below)
#$ -t 1-48

# This job is created by using an array job. This means that we index into the noise amplitude and noise correlation using the single index
# SGE_TASK_ID. This ID range, listed above, must span between 1-num_jobs where num_jobs=num_noise_corr*num_noise_amp. These numbers 
# currently need to match what is in the python code. I can probably make this more robust, but this is how I'm doing it for now.
# BE CAREFUL!

savedir=$1"/"
# Get output directory
if [ -z "$1" ]
  then
    echo "No directory specified. Using output/"
    savedir="output/"
fi

mkdir -p $savedir

echo "Task ID is $SGE_TASK_ID"

num_bw=8

# i is noise_amp and j is noise_corr
SGE_TASK_ID_MINUS_ONE=$((SGE_TASK_ID -1))
i=$((SGE_TASK_ID_MINUS_ONE / num_bw))
j=$((SGE_TASK_ID_MINUS_ONE % num_bw))

cp MultiAxisDephasingPMD.py $savedir"/"
cp SurfaceXSyndrome.py $savedir"/"

echo "python \"SurfaceXSyndrome.py $i $j $savedir\" -logfile "$savedir"logfile_"$i"_"$j".out"
python SurfaceXSyndrome.py $i $j $savedir -logfile $savedir"logfile_"$i"_"$j".out" 1> $savedir"surface_xsyn_"$i"_"$j".stdout" 2> $savedir"surface_xsyn_"$i"_"$j".stderr"  
