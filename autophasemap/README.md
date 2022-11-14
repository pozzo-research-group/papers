## Installing environment on Hyak

```bash
conda env create --prefix ~/kiranvad/envs/elastic_kmeans --file environment.yml
```

And activate it using:

```bash
conda activate elastic_kmeans
```

You need to install the following:

```bash
pip install numpy Cython cffi
```

Load the gcc module
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