import numpy as np
import matplotlib.pyplot as plt


def avg_dose_response(X, T,  Y, model, func_create_predictor):

    avg_dose_response = []
    td = np.arange(min(np.array(T)), max(np.array(T)) + 0.1, 0.1)

    for t in td:
        predictor = func_create_predictor(X, np.tile(t, len(X)))
        outcomes_pred = model.predict(predictor)
        avg_dose_response.append(np.mean(outcomes_pred))

    fig = plt.figure()
    plt.plot(td, avg_dose_response)
    plt.show()  # slope looks like what we expected.












