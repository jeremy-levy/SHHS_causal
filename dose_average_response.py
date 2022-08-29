import numpy as np
import matplotlib.pyplot as plt


def avg_dose_response(X, min_T, max_T, delta_t, model, plot_figure=False):

    avg_dose_response_list = []
    td = np.arange(min_T, max_T + delta_t, delta_t)

    for t in td:
        predictor = np.column_stack([X, np.tile(t, len(X))])
        outcomes_pred = model.predict(predictor)
        avg_dose_response_list.append(np.mean(outcomes_pred))

    if plot_figure is True:
        plt.figure()
        plt.plot(td, avg_dose_response_list)
        plt.show()  # slope looks like what we expected.

    return np.array(avg_dose_response_list)
