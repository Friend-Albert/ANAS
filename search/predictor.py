import json
import argparse
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


class Predictor:
    def __init__(self, path, n_encoder, n_decoder):
        self.path = path
        self.n_input = 2 * (n_encoder + n_decoder - 2)
        self.model = xgb.XGBRegressor(max_depth=12, learning_rate=0.1, n_estimators=1000, subsample=0.9)

    def train(self):
        json_datas, datas = [], []
        try:
            with open(self.path, 'r') as f:
                data = f.readlines()
                for i in data:
                    json_datas.append(json.loads(i))
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.path}")
            return
            
        for data in json_datas:
            tmp = data['cell'] + [data['ret']]
            datas.append(tmp)
        datas = np.array(datas)
        X = datas[:, 0:self.n_input]
        Y = datas[:, self.n_input]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        mae_loss = np.mean(abs(y_test - y_pred))
        print(f"XGBoost Regressor MAELoss is {mae_loss}")

    def predict(self, cell: list):
        cell = np.array([cell])
        return self.model.predict(cell)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and run the XGBoost performance predictor.")
    parser.add_argument('--path', type=str, default='../logs/cell_data.json', help='Path to the cell data log file.')
    parser.add_argument('--n_encoder', type=int, default=8, help='Number of nodes in the encoder.')
    parser.add_argument('--n_decoder', type=int, default=8, help='Number of nodes in the decoder.')
    args = parser.parse_args()

    p = Predictor(args.path, args.n_encoder, args.n_decoder)
    p.train()
    
    # Example prediction
    print("\nRunning example prediction:")
    l = [2, 1, 1, 0, 1, 0, 0, 0, 1, 0, 2, 0, 2, 1, 0, 0, 2, 0, 4, 0, 5, 0, 0, 2, 3, 1, 4, 3]
    if len(l) == 2 * (args.n_encoder + args.n_decoder - 2):
        print(f"Predicted performance for example cell: {p.predict(l)}")
    else:
        print(f"Example cell length does not match the required input size for the model: {2 * (args.n_encoder + args.n_decoder - 2)}")
