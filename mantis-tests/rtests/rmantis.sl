#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --mem=40G
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:3090:1                # If you want to use a GPU <-- DO NOT FORGET !
#SBATCH --mail-type=ALL
#SBATCH --mail-user=h.topazian@utah.edu
#SBATCH --output=$HOME/rmantis-%j.out
#SBATCH --error=$HOME/rmantis-%j.err
#SBATCH -J rmantis-job

# Exit on error, undefined vars, pipe failures
set -euo pipefail   
module purge

# Environmental Variables
export SCRATCHDIR=/scratch/general/vast/$USER/$SLURM_JOBID
export WORKDIR=$HOME/RESPIRATORYforecastUT/HPC
export INPUTFILE=mantis_test.R
export OUTPUTFILE=mantis_test.out


printf "Jobid: %s\n" "$SLURM_JOBID"
printf "  Started at : %s\n"  "$(date)"
printf "  Hostname   : %s\n"  "$(hostname)"
printf "  nvidia-smi :\n%s\n" "$(nvidia-smi)"
printf "  R version  : %s\n"  "$(R --version | head -1)"
printf "  R exe.     : %s\n"  "$(which R)"


# A.Create scratch directory - cp WORKDIR material to scratch
mkdir -p $SCRATCHDIR && cd $SCRATCHDIR 
cp -pR $WORKDIR/*  .

# B.Run simulation
Rscript $INPUTFILE >& $OUTPUTFILE

RSCRIPT_EXIT=$?

if [ $RSCRIPT_EXIT -ne 0 ]; then
    echo "ERROR: Rscript exited with $RSCRIPT_EXIT — not copying back." >&2
    trap - EXIT
    exit $RSCRIPT_EXIT
fi


# C.Copy back results
cd $WORKDIR
cp -pR $SCRATCHDIR/* .

# D.Clean up scratch dir.
rm -rf $SCRATCHDIR

printf "\n  Ended at : %s\n"  "$(date)"

