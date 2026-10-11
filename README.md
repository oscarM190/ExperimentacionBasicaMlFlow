Execute the commands:
1.- virtualenv venv3.13 --python=python3.14.7
2.- source venv3.13/bin/activate
3.- cd experimentation_mlflow
4.- pip3 install -r requirements.txt
5.- python3 src/main.py --dummyStrategy 'most_frequent' --penalty 'l2' --random_state 7 --solver 'saga'


Nota: El comando 5 puedes varias los hiperparametros para el modelo dummy: dummyStrategy; Para el Modelo de regresion logistica: penalty, random_state y solver
