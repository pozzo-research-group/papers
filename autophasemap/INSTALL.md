The installation instruction are below for a cluster computer but there are the same for a local computer.
The main difference is that on a cluster computer you need to explicitly load the module `gcc` which would be readily available on your Mac/Linux OS systems.
For windows, install the gcc might be tricky. Our suggestion would be to try using [VSCode](https://code.visualstudio.com/docs/cpp/config-mingw).

## Installing environment on cluster computer

Load the gcc module if you are using this on a cluster computer otherwise move to the next step affter making sure the C/C++ compliers are installed.
```bash
module load gcc
```

```bash
conda env create --prefix ~/ENV_LOCATION/autophasemap --file environment.yml
```

And activate it using:

```bash
conda activate autophasemap
```

Finally, the `autophasemap` can be installed using:

```bash
pip install -e .
```
