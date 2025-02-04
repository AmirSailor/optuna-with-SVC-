# optuna-with-SVC
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import optuna


def objective(trial):
    C = trial.suggest_int('C', 1, 50)
    kernel = trial.suggest_categorical('kernel', ['poly', 'rbf', 'sigmoid'])
    tol = trial.suggest_float('tol', 0.05, 0.1)
    decision_function_shape = trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr'])
    random_state = trial.suggest_int('random_state', 1, 1000)
    shrinking = trial.suggest_categorical('shrinking', [True, False])
    gamma = trial.suggest_float('gamma', 0.0001, 1)

    model = SVC(
        C=C,
        kernel=kernel,
        tol=tol,
        decision_function_shape=decision_function_shape,
        random_state=random_state,
        shrinking=shrinking,
        gamma=gamma
    )

    score = cross_val_score(model, X_train, y_train, cv=3)
    accuracy = score.mean()
    return accuracy

search_space = {
    'C': [1, 5, 15, 25, 35, 40, 50],
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'tol': [0.05, 0.06, 0.07, 0.08, 0.1],
    'decision_function_shape': ['ovo', 'ovr'],
    'random_state': [10, 50, 150, 350, 550, 750, 950, 1000],
    'shrinking': [True, False],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 0.5, 1]
}

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.GridSampler(search_space)
)
study.optimize(objective)
