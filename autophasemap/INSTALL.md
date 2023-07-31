The installation instruction are below for a cluster computer but there are the same for a local computer.
The main difference is that on a cluster computer you need to explicitly load the module `gcc` which would be readily available on your Mac/Windows/Linux OS systems.

## Installing environment on cluster computer

```bash
conda env create --prefix ~/<your_username>/envs/elastic_kmeans --file environment.yml
```

And activate it using:

```bash
conda activate elastic_kmeans
```

You need to install the following:

```bash
pip install numpy Cython cffi
```

Load the gcc module if you are using this on a cluster computer otherwise move to the next step.
```bash
module load gcc
```

And then install the warping package using git : 

```bash
pip install git+https://github.com/kiranvad/warping.git
```

Finally, the `autophasemap` can be installed using:

```bash
pip install -e .
```
