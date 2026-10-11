from BreastCancerModel import BreastCancerModel
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Este modelo acepta como parametros: path, dummyStrategy (para modelo dummy), penalty (para regresion logistica), random_state (para regresion lineal) y solver (para regresion lineal)"
    )

    # Definición de parámetros
    parser.add_argument(
        "--path", type=str, required=False, help="Ruta donde estan los datos", default="data/raw/breast_cancer.csv"
    )
    parser.add_argument(
        "--dummyStrategy", type=str, required=False, help="Estrategia del modelo dummy", default="stratified"
    )
    parser.add_argument(
        "--penalty", type=str, required=False, help="Penalizacion de la regresion logistica", default=None
    )

    parser.add_argument(
        "--random_state", type=int, required=False, help="random_state de la regresion logistica", default=1
    )

    parser.add_argument(
        "--solver", type=str, required=False, help="solver de la regresion logistica", default='lbfgs'
    )
    
    args = parser.parse_args()
    print(args.path)
    breastCancerModel = BreastCancerModel(args.path)
    breastCancerModel.load_data()
    breastCancerModel.preprocess_data()
    breastCancerModel.train_dummy_model(args.dummyStrategy, args.random_state)
    breastCancerModel.train_lr_model(args.random_state, args.penalty, args.solver)

if __name__ == "__main__":
    main()
    