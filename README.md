### Double crystal interferometer scripts.

Tested for Python 3.6+.

Required for some packages:
```sh
sudo apt-get install python3-dev 
sudo apt-get install libevent-dev
```

How to set up an environment:
```sh
# Install environment.
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt
```

How to run:
```sh
python3 double_crystal_scheme.py --d=0.5 --L1=0.1 --save_path=path/
```

