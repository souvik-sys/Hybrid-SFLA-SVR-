import numpy as np
import joblib
import math
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class HybridSFLASVR:
    def __init__(self, param_map, scorer='neg_mean_squared_error', cv=3, random_state=None):
        self.param_map = param_map
        self.scorer = scorer
        self.cv = cv
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None

    def rastrigin(self, x, A=10):
        x = np.array(list(x.values()))
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def levy_flight(self, beta=1.5, size=1):
        sigma_u = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(size) * sigma_u
        v = np.random.randn(size)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def random_init(self, bounds):
        return {k: np.random.uniform(v[0], v[1]) for k, v in bounds.items()}

    def ensure_bounds(self, vec, bounds):
        for k in vec.keys():
            low, high = bounds[k]
            vec[k] = np.clip(vec[k], low, high)
        return vec

    def evaluate(self, params, X, y):
        rastrigin_term = self.rastrigin(params)  
        model = SVR(**params)
        score = cross_val_score(model, X, y, scoring=self.scorer, cv=self.cv).mean()
        combined = score - 0.01 * rastrigin_term
        return combined

    def fit(self, X, y, frogs=30, mplx_no=6, mplx_iters=10, solun_iters=30, alpha=0.01, beta=1.5):
        np.random.seed(self.random_state)
        pop = [self.random_init(self.param_map) for _ in range(frogs)]
        scores = [self.evaluate(p, X, y) for p in pop]
        best_idx = np.argmax(scores)
        global_best = pop[best_idx]
        global_best_score = scores[best_idx]

        m = frogs // mplx_no
        for _ in range(solun_iters):
            idxs = np.argsort(scores)[::-1]
            pop = [pop[i] for i in idxs]
            scores = [scores[i] for i in idxs]
            memeplexes = [pop[i*m:(i+1)*m] for i in range(mplx_no)]

            for memeplex in memeplexes:
                for _ in range(mplx_iters):
                    best = memeplex[0]
                    worst = memeplex[-1]
                    step = alpha * self.levy_flight(beta, len(self.param_map))
                    new_worst = {k: worst[k] + s for k, s in zip(worst.keys(), step)}
                    new_worst = self.ensure_bounds(new_worst, self.param_map)
                    new_score = self.evaluate(new_worst, X, y)
                    if new_score > self.evaluate(worst, X, y):
                        memeplex[-1] = new_worst

            pop = [frog for memeplex in memeplexes for frog in memeplex]
            scores = [self.evaluate(p, X, y) for p in pop]
            best_idx = np.argmax(scores)
            if scores[best_idx] > global_best_score:
                global_best = pop[best_idx]
                global_best_score = scores[best_idx]

        self.best_params_ = global_best
        self.best_score_ = global_best_score
        self.best_model_ = SVR(**global_best)
        self.best_model_.fit(X, y)

    def get_best_params(self):
        return self.best_params_

    def get_best_model(self):
        return self.best_model_

    def predict_and_compare(self, X_test, y_test):
        y_pred = self.best_model_.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
        return y_pred, metrics

    def save(self, filename):
        joblib.dump({
            'best_model': self.best_model_,
            'best_params': self.best_params_,
            'best_score': self.best_score_
        }, filename)

    @staticmethod
    def load(filename):
        data = joblib.load(filename)
        return data['best_model'], data['best_params'], data['best_score']

def run_optimizer(x_train, y_train, x_test, y_test, param_map, frogs=30, mplx_no=6, mplx_iters=10, solun_iters=30):
    opt = HybridSFLASVR(param_map, scorer='neg_mean_squared_error', cv=3, random_state=42)
    opt.fit(x_train, y_train, frogs=frogs, mplx_no=mplx_no, mplx_iters=mplx_iters, solun_iters=solun_iters)
    y_pred, metrics = opt.predict_and_compare(x_test, y_test)
    print("\nBest Parameters:", opt.get_best_params())
    print("Best CV Score:", opt.best_score_)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    return opt
