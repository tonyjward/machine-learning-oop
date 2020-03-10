# Machine Learning Object Oriented Programming
There are two goals of this project 
1) To practice object oriented programming in Python
2) To improve Machine Learning skills by implementing algorithms from scratch

We start with the simplest of the machine learning algorithms - Linear Regression. Three optimisers have been coded up
1) [Ordinary Least Squares] (https://github.com/tonyjward/machine-learning-oop/blob/master/twlearn/LinearRegression.py)
2) [Gradient Descent](https://github.com/tonyjward/machine-learning-oop/blob/master/twlearn/GradientDescent.py)
3) [Particle Swarm](https://github.com/tonyjward/machine-learning-oop/blob/master/twlearn/ParticleSwarm.py)

Particle Swarm Optimisation (PSO) allows us to optimise custom loss functions. The nice thing here is unlike gradient descent type algorithms, the loss function does not have to be differntiable. In [this notebook](https://github.com/tonyjward/machine-learning-oop/blob/master/notebooks/OLS-vs-PSO.ipynb) I demonstrate how we can take advantage of this by optimising a range of custom loss functions. I also test whether the resulting models are significantly different from each other, using the boston housing dataset.

#### -- Project Status: Under Development

### Methods Used
* Object Orientated Programming
* Optimisation (OLS, Gradient Descent, Particle Swarm)

### Technologies
* Python

### Tests
Tests can be run from the main directory using
```
python -m unittest discover
```

## Contact
* tony@statcore.co.uk
