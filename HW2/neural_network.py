import mlrose
import pandas as pd
from sklearn.metrics import accuracy_score

if __name__=="__main__":
    df_eeg = pd.read_csv("eeg_dataset.csv")

    y = df_eeg['y']
    X = df_eeg.drop(columns=['y'])

    model_hc = mlrose.NeuralNetwork(hidden_nodes = [1024,512,256,32], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 random_state = 3)
    model_hc.fit(X, y)

    y_pred = model_hc.predict(X)

    model_hc.score

    print(accuracy_score(y, y_pred))