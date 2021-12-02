python3 -m venv env
source env/bin/activate
mkdir ext
cd ext 
pip install 'numpy<=1.20' scipy matplotlib
git clone https://gitlab.com/wiechapeter/pyGDM2.git
pip install ./pyGDM2
git clone https://github.com/pytorch/botorch.git
pip install ./botorch
git clone https://github.com/kiranvad/geomstats.git
pip install ./geomstats
echo "Installation complete!!!"

