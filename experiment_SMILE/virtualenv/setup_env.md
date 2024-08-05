what worked best:

# at Kispi stimulus computer (windows)
problems with running python from Caroline's account. 
--> Created "standalone python" (as copy from preexisting Psychopy installation - python 3.8)
* installed from requirements.txt (with some removements (#) of mac-specific packages, mostly 'pyobv-...', but just look for error)
* installed exptools 2 via getting folder from git (from gilles86), and then pip install (git command not on computer!)
--> go into experiment folder, call "python" via= 'C:\"Program Files"\Psychopy202302\python.exe' (weird "" in pathname)

# my computer 
## psychopy2
cd ~/Desktop/codes../psychopy/
conda create --name psychopy2 --file psychopy2.txt
conda activate psychopy2
pip install git+https://github.com/Gilles86/exptools2 --no-deps

## psychopy
also works, bit of a hassle (see at bottom)
-------------------------------------------------------
* run `pip install -r requirements.txt` until 

* `gevent==20.5.2` error
pip install pip setuptools wheel Cython==3.0.0a10
pip install gevent==20.9.0 --no-build-isolation

* run `pip install -r requirements_.txt`

* error `pip install tables=3.6.1`

brew install hdf5

* conda install pytables
* pip install psychopy==2020.1.2 

conda uninstall numpy
conda install numpy
* 
* 
* y


* conda create -n psychopy_env
pip install -r requirements.txt (conda --> packages are not available)

      Cython.Compiler.Errors.CompileError: src/gevent/greenlet.py

❯ conda env create -n psychopy_env2 -f psychopy-env.yml
Collecting package metadata (repodata.json): done
Solving environment: failed

ResolvePackageNotFound:
  - python=3.6



conda install -n <name enviornment> ipykernel --update-deps --force-reinstall



conda create -n psychopy psychopy
conda activate psychopy
pip install git+https://github.com/Gilles86/exptools2 --no-deps

pip install ipython
-> `from psychopy import visual, core `
Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.

conda uninstall numpy
conda install numpy

ipython --> `ModuleNotFoundError: No module named 'six'`

conda install psychopy