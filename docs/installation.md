# Installation

## Conda
Prerequisite: `conda==24.9.2`, `Ubuntu==24.04.1`
```bash
conda env create -f .env/environment.yml -n venv
```
```bash
conda activate venv
```
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118  --force
```
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Virtualenv

### Linux
```bash
apt install python3-virtualenv
```
```bash
virtualenv venv --python=python3.10
```
```bash
source venv/bin/activate
```
```bash
pip install --upgrade pip
```
```bash
pip install -r .env/requirements.txt
```
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Window
```bash
pip install virtualenv
```
```bash
virtualenv venv --python=python3.10
```
```bash
.\venv\Scripts\activate
```
```bash
pip install --upgrade pip
```
```bash
pip install -r .env\requirements.txt
```
```bash
python -c "import torch; print(torch.cuda.is_available())"
```