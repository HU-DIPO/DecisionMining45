# Decision Mining Backend

This repository can either be installed and used as a Python module or as a decision mining API.

## Installing as a module

```shell
git clone https://github.com/HU-DIPO/DecisionMining45.git
cd DecisionMining45\\back-end
pip install .
```

Or as a developer:

```shell
git clone https://github.com/HU-DIPO/DecisionMining45.git
cd DecisionMining45\\back-end
pip install .[dev]
```

After this, simply `import decision_mining` in your own Python code.

## Running API individually

As the name implies, the backend can also be used together with the frontend. It can also be run individually, either locally or through a docker container.

### Running with Docker

```shell
git clone https://github.com/HU-DIPO/DecisionMining45.git
cd DecisionMining45\\back-end
docker build -t dm45/back:v1 .
docker run -p 5000:5000 dm45/back:v1
```

And then go to `localhost:5000`.

### Running with Flask

Repeat instructions from `Installing as a module`, then:

```shell
flask run
```

And then go to `localhost:5000`.

## Generate the documentation

To generate the documentation of this project we use Sphinx.

After installing DM45 as dev, do:

```shell
.DM45\\Backend\\srcdocs\\make.bat html
```
