# Fast-CMA

__Organization of the repository__


1) cmalight.py -> CMA Light code, from the pre-print https://arxiv.org/abs/2307.15775
2) FCMA.py -> Fast-CMA
3) main.py -> main file
4) network.py -> Wrapper for the different models
5) utils.py -> Utility functions

To execute the attached code, navigate to the directory containing the Docker image and build it using the following command:


__docker build -t cuda12.1.0_python3.8_pytorch2.1.0 .__


After building the image, modify the paths inside \texttt{run\_single\_train.sh}. 
Subsequently, any experiments can be run by executing the following script from the command prompt:

__sh run_single_train.sh__


Furthermore, the network architecture, optimizer, and dataset can be modified by appending the respective arguments to the script invocation in the command prompt. For instance, if one desires to run the Resnet18 architecture with FCAM optimization on the CIFAR-10 dataset, the command would be as follows:

__sh run_single_train.sh resnet18 fcma cifar10__
