#!/bin/bash
#SBATCH --partition titanx-short
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 10GB
#SBATCH --time 4:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output slurm-%J.out 
#SBATCH -e slurm-%J.err
#SBATCH --mail-type END
#SBATCH --mail-user huiwei@umass.edu


# Thank you, Yale Center for Research Computing, for this script
# modified for use on BigPurple in NYU Langone Medical Center by Paul Glick

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
login_node=gypsum.cs.umass.edu

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -L ${port}:${node}:${port} ${user}@${login_node} 

For more info and how to connect from windows, 
   see research.computing.yale.edu/jupyter-nb
Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${login_node}
SSH login: $user
SSH port: 22
Use a Browser on your local machine to go to:
http://localhost:${port}  (prefix w/ https:// if using password)
Use the token string from the URL printed below and add it to the URL above

Please paste this to your browser:
http://localhost:${port}/?token=$(pwd)
"

# load modules or conda environments here
# e.g. farnam:
# module load Python/2.7.13-foss-2016b 
# module load python/gpu/3.6.5
# DON'T USE ADDRESS BELOW. 
# DO USE TOKEN BELOW
source activate mldl
sleep 1
jupyter-notebook --no-browser --port=${port} --ip=${node} --notebook-dir=$(pwd)
