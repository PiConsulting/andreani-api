# Andreani API

## Setup
python 3.7 >  is needed.
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## How to run
```
cd src
uvicorn main:app --reload
```

or 
### Docker
```
docker build -t andreani-api .
docker run -p 8000:8000 andreani-api
```


### Authentication
Username = test

Password = 123

*else if look in `auth.py`*

## Docs
`http://127.0.0.1:8000/docs`