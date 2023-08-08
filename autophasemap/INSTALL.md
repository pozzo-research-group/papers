The installation instruction are below for a cluster computer but there are the same for a local computer.
The main difference is that on a cluster computer you need to explicitly load the module `gcc` which would be readily available on your Mac/Linux OS systems.
For windows, install the gcc might be tricky. Our suggestion would be to try using [VSCode](https://code.visualstudio.com/docs/cpp/config-mingw).

## Installation instructions

First, clone the repository using git:
```bash
git clone https://github.com/pozzo-research-group/papers.git
```

or if you want to install this code with version tag on release, use the following:
```bash 
git clone --depth 1 --branch v1.0.0-autophasemap https://github.com/pozzo-research-group/papers.git
```

Load the gcc module if you are using this on a cluster computer otherwise move to the next step affter making sure the C/C++ compliers are installed.
```bash
module load gcc
```

```bash
conda env create --prefix /ENV_LOCATION/autophasemap --file environment.yml
```

If you used a specific location that is not already in the conda environments path, run the following command to add it to the path:
```bash
conda config --append envs_dirs ENV_LOCATION
```

And activate it using:

```bash
conda activate autophasemap
```

You now have to install the warping function using the following:
```bash
pip install git+https://github.com/kiranvad/warping.git 
``` 

or if you want to install this code with version tag on release, use the following:
```bash 
pip install git+https://github.com/kiranvad/warping.git@v0.1.0
```

This should work if you have the C/C++ compliter and Cython package installed using the above instructions.

Finally, the `autophasemap` can be installed using after changing the directory into the autophasemap:

```bash
cd autophasemap
pip install -e .
```
