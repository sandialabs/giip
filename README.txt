This software library relies on pytorch, while the example script provided requires h5py, networkx, and numpy in addition to torch. Advanced users can install Jupyter and execute the visualization notebook with the example script, but I've rendered the visualization as html documents for reviewers' convenience.

I suggest setting up a conda environment to sandbox this installation. In theory this could work in Windows, but I have only tested it in Linux and WSL. So, if you don't already have Miniconda or Anaconda installed, do that now.

Create a conda environment with the required prerequisite python packages:
conda create --name GIIP python=3.9 h5py networkx numpy pytorch

Activate the environment
conda activate GIIP

Now, let's assume that you've unzipped the GIIP software package into some directory, resulting in a directory structure along these lines:

SOMEDIRECTORY\
|- giip\
	|- assets\
	|- examples\
	|- GIIP\
	|- CHANGELOG.md
	|- README.txt
	|- .gitignore
	etc

You can execute the example script as follows:
PYTHONPATH=SOMEDIRECTORY/giip-b-release-0.2.0 python SOMEDIRECTORY/giip-b-release-0.2.0/examples/example01-metal2d/main.py
You may need to modify the line that reads device = 'gpu' to device = 'cpu' if your torch installation didn't include GPU support.
At the top of the example script you can pick between a "quick but uninteresting example" that takes less than a minute to run on my laptop, and a "slower but more interesting" example that takes about 20 minutes on my laptop.
You can safely ignore a "UserWarning" about creating a tensor from numpy ndarrays.
If the code runs successfully, the result is an hdf file in the example directory that contains a matrix of GIIP distances associated with the example. If you're motivated to do so, you can use that matrix of distances in any implementation of diffusion maps or agglomerative clustering that you like.


Note that example02-mg3 only contains the underlying data for the MG portion of this manuscript. GIIP processing in three dimensions is far more onerous and storage-intensive. Reach out to the corresponding author for help optimizing the performance of your distribution of the GIIP library before you attempt to reproduce the results in that portion of the paper.

Note also that this distribution of the GIIP code only contains three-dimensional samplings of SO(3) down to three degrees of resolution. This is to save on disk space. The final distribution of GIIP will include samplings down to one degree of resolution; you can also generate your own coverings using pmla/hyperspherical-coverings.