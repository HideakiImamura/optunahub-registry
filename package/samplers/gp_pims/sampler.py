from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from collections.abc import Callable
import time
from typing import Any

import gpytorch
import numpy as np
import optuna
from optuna.distributions import FloatDistribution
import optunahub
from scipy import optimize
from scipy.stats import qmc
import torch


class RFM_RBF:
    """
    rbf(gaussian) kernel k(x, y) = variance * exp(- 0.5 * ||x - y||_2^2 / lengthscale**2)
    """

    def __init__(
        self, lengthscales: np.ndarray, input_dim: int, variance: float = 1, basis_dim: int = 1000
    ) -> None:
        self.basis_dim = basis_dim
        self.std = np.sqrt(variance)
        self.random_weights = (1 / np.atleast_2d(lengthscales)) * np.random.normal(
            size=(basis_dim, input_dim)
        )
        self.random_offset = np.random.uniform(0, 2 * np.pi, size=basis_dim)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        X_transform = X.dot(self.random_weights.T) + self.random_offset
        X_transform = self.std * np.sqrt(2 / self.basis_dim) * np.cos(X_transform)
        return X_transform

    def transform_grad(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        X_transform_grad = X.dot(self.random_weights.T) + self.random_offset
        X_transform_grad = (
            -self.std
            * np.sqrt(2 / self.basis_dim)
            * np.sin(X_transform_grad)
            * self.random_weights.T
        )
        return X_transform_grad


def minimize(
    func: Callable,
    start_points: np.ndarray,
    bounds: np.ndarray,
    jac: Callable | None = None,
    first_ftol: float = 1e-1,
    second_ftol: float = 1e-2,
) -> tuple[np.ndarray, float]:
    x = np.copy(start_points)
    func_values = list()
    for i in range(np.shape(x)[0]):
        res = optimize.minimize(
            func, x0=x[i], bounds=bounds, method="L-BFGS-B", options={"ftol": first_ftol}, jac=jac
        )
        func_values.append(res["fun"])
        x[i] = res["x"]

    if second_ftol < first_ftol:
        f_min = np.min(func_values)
        f_max = np.max(func_values)
        index = np.where(func_values <= f_min + (f_max - f_min) * 1e-1)[0]

        for i in index:
            res = optimize.minimize(
                func,
                x0=x[i],
                bounds=bounds,
                method="L-BFGS-B",
                options={"ftol": second_ftol},
                jac=jac,
            )
            func_values[i] = res["fun"]
            x[i] = res["x"]

    min_index = np.argmin(func_values)
    return x[min_index], func_values[min_index]


class GP_model(gpytorch.models.ExactGP):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        noise_var: float = 1e-4,
    ) -> None:
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.tensor([noise_var])
        )
        super().__init__(
            train_inputs=torch.from_numpy(X),
            train_targets=torch.from_numpy(Y),
            likelihood=likelihood,
        )
        assert len(self.train_inputs) == 1

        lower_bound = min(0.9 * Y.mean(), 1.1 * Y.mean())
        upper_bound = max(0.9 * Y.mean(), 1.1 * Y.mean())
        self.mean_module = gpytorch.means.ConstantMean(
            constant_constraint=gpytorch.constraints.Interval(lower_bound, upper_bound)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.Interval(1e-6, 1e6)
            ),
            outputscale_constraint=gpytorch.constraints.Interval(1e-6, 1e6),
        )

        self.my_optimize()

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def my_optimize(self, training_iter: int = 50) -> None:
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.1
        )  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(training_iter):
            print(f"training iteration: {i}")
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(self.train_inputs[0])
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_targets)
            loss.backward()
            optimizer.step()

    @property
    def X(self) -> np.ndarray:
        return self.train_inputs[0].numpy()

    @property
    def Y(self) -> np.ndarray:
        return self.train_targets.numpy()

    def add_XY(self, X: np.ndarray, Y: np.ndarray) -> None:
        new_X = torch.cat((self.X, torch.from_numpy(X)))
        new_Y = torch.cat((self.Y, torch.from_numpy(Y)))
        self.set_train_data(new_X, new_Y)
        lower_bound = min(0.9 * new_Y.mean(), 1.1 * new_Y.mean())
        upper_bound = max(0.9 * new_Y.mean(), 1.1 * new_Y.mean())
        self.mean_module = gpytorch.means.ConstantMean(
            constant_constraint=gpytorch.constraints.Interval(lower_bound, upper_bound)
        )

    def predict_noiseless(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        test_x = torch.from_numpy(X)
        observed_pred = self(test_x)
        mean = observed_pred.mean.numpy()
        var = observed_pred.variance.numpy()
        return mean, var

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        test_x = torch.from_numpy(X)
        observed_pred = self.likelihood(self(test_x))
        mean = observed_pred.mean.numpy()
        var = observed_pred.variance.numpy()
        return mean, var


class BO_core(object):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        bounds: np.ndarray,
        optimize: bool = True,
    ) -> None:
        self.GPmodel = GP_model(X, Y)
        self.y_max = np.max(Y)
        self.unique_X = np.unique(X, axis=0)
        self.input_dim = np.shape(X)[1]
        self.bounds = bounds
        self.bounds_list = bounds.T.tolist()
        self.sampling_num = 10
        self.inference_point = None
        self.top_number = 50
        self.preprocessing_time = 0.0
        self.max_inputs = None

    def update(self, X: np.ndarray, Y: np.ndarray, optimize: bool = False) -> None:
        self.GPmodel.add_XY(X, Y)
        if optimize:
            self.GPmodel.my_optimize()
        else:
            self.GPmodel.train()
            self.GPmodel.likelihood.train()

        self.y_max = np.max(self.GPmodel.Y)
        self.unique_X = np.unique(self.GPmodel.X, axis=0)

    @abstractmethod
    def acq(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def next_input(self) -> np.ndarray:
        pass

    def upper_bound(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        return mean + 5.0 * np.sqrt(var)

    def sampling_RFM(self, pool_X: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        # 基底をサンプリング, n_compenontsは基底数, random_stateは基底サンプリング時のseed的なの
        basis_dim = 500 + np.shape(self.GPmodel.X)[0]
        self.rbf_features = RFM_RBF(
            lengthscales=self.GPmodel.covar_module.base_kernel.lengthscale.detach().numpy(),
            input_dim=self.input_dim,
            basis_dim=basis_dim,
            variance=self.GPmodel.covar_module.outputscale.detach().numpy(),
        )
        X_train_features = self.rbf_features.transform(self.GPmodel.X)

        max_sample = np.zeros(self.sampling_num)
        max_inputs = list()

        A_inv = np.linalg.inv(
            (X_train_features.T).dot(X_train_features)
            + np.eye(self.rbf_features.basis_dim) * self.GPmodel.likelihood.noise.detach().numpy()
        )
        weights_mean = A_inv.dot(X_train_features.T).dot((self.GPmodel.Y - self.GPmodel.Y.mean()))
        weights_var = A_inv * self.GPmodel.likelihood.noise.detach().numpy()

        try:
            L = np.linalg.cholesky(weights_var)
        except np.linalg.LinAlgError as e:
            print("In RFM-based sampling,", e)
            L = np.linalg.cholesky(weights_var + 1e-5 * np.eye(np.shape(weights_var)[0]))

        # 自分で多次元正規乱数のサンプリング
        standard_normal_rvs = np.random.normal(
            0, 1, size=(np.size(weights_mean), self.sampling_num)
        )
        self.weights_sample = np.c_[weights_mean] + L.dot(standard_normal_rvs)

        if pool_X is None:
            num_start = 100 * self.input_dim

            sampler = qmc.Halton(d=self.input_dim, scramble=False)
            sample = sampler.random(n=num_start)
            x0s = qmc.scale(sample, self.bounds[0], self.bounds[1])

            if np.shape(self.unique_X)[0] <= self.top_number:
                x0s = np.r_[x0s, self.unique_X]
            else:
                mean, _ = self.GPmodel.predict(self.unique_X)
                mean = mean.ravel()
                top_idx = np.argpartition(mean, -self.top_number)[-self.top_number :]
                x0s = np.r_[x0s, self.unique_X[top_idx]]
        else:
            if np.size(pool_X[(self.upper_bound(pool_X) >= self.y_max).ravel()]) > 0:
                pool_X = pool_X[(self.upper_bound(pool_X) >= self.y_max).ravel()]

        for j in range(self.sampling_num):

            def BLR(x: np.ndarray) -> np.ndarray:
                X_features = self.rbf_features.transform(x)
                sampled_value = X_features.dot(np.c_[self.weights_sample[:, j]])
                return -(sampled_value + self.GPmodel.Y.mean()).ravel()

            def BLR_gradients(x: np.ndarray) -> np.ndarray:
                X_features = self.rbf_features.transform_grad(x)
                sampled_value = X_features.dot(np.c_[self.weights_sample[:, j]])
                return -(sampled_value).ravel()

            if pool_X is None:
                f_min = np.inf
                x_min = x0s[0]

                x_min, f_min = minimize(BLR, x0s, self.bounds_list, jac=BLR_gradients)
                max_sample[j] = -1 * f_min
                max_inputs.append(x_min)
            else:
                pool_Y = BLR(pool_X)
                min_index = np.argmin(pool_Y)
                max_sample[j] = -1 * pool_Y[min_index]
                max_inputs.append(pool_X[min_index])

        return max_sample, np.array(max_inputs)

    def sample_path(self, X: np.ndarray) -> np.ndarray:
        """
        Return the corresponding value of the sample_path using sampling_num RFMs for the input set X.

        Parameter
        -----------------------
        X: numpy array
            inputs (N \times input_dim)

        Return
        -----------------------
        sampled_outputs: numpy array
            sample_path f_s(X) (N \times sampling_num)
        """
        X_features = self.rbf_features.transform(X)
        sampled_outputs = X_features.dot(np.c_[self.weights_sample]) + self.GPmodel.Y.mean()
        return sampled_outputs


class BO(BO_core):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        bounds: np.ndarray,
        optimize: bool = True,
    ) -> None:
        super().__init__(X, Y, bounds, optimize=optimize)

    def next_input(self) -> np.ndarray:
        num_start = 100 * self.input_dim

        sampler = qmc.Halton(d=self.input_dim, scramble=False)
        sample = sampler.random(n=num_start)
        x0s = qmc.scale(sample, self.bounds[0], self.bounds[1])

        x0s = x0s[(self.upper_bound(x0s) >= self.y_max).ravel()]
        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean, _ = self.GPmodel.predict(self.unique_X)
            mean = mean.ravel()
            top_idx = np.argpartition(mean, -self.top_number)[-self.top_number :]
            x0s = np.r_[x0s, self.unique_X[top_idx]]

        f_min = np.inf
        x_min = x0s[0]
        if self.max_inputs is not None:
            x0s = np.r_[x0s, self.max_inputs]

        x_min, f_min = minimize(self.acq, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
        print("optimized acquisition function value:", -1 * f_min)
        return np.atleast_2d(x_min)


class PI_from_MaxSample(BO):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        bounds: np.ndarray,
        pool_X: np.ndarray | None = None,
        optimize: bool = True,
    ) -> None:
        super().__init__(X, Y, bounds, optimize=optimize)

        self.input_dim = np.shape(X)[1]
        self.pool_X = pool_X
        self.sampling_num = 1

        start = time.time()
        self.maximums, self.max_inputs = self.sampling_RFM(pool_X)
        self.preprocessing_time = time.time() - start
        print("sampled maximums:", self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def update(self, X: np.ndarray, Y: np.ndarray, optimize: bool = False) -> None:
        super().update(X, Y, optimize=optimize)
        start = time.time()
        self.maximums, self.max_inputs = self.sampling_RFM(self.pool_X)
        self.preprocessing_time = time.time() - start
        print("sampled maximums:", self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def acq(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)

        return ((self.maximums - mean) / std).ravel()


SimpleBaseSampler = optunahub.load_module("samplers/simple").SimpleBaseSampler


class PIMSSampler(SimpleBaseSampler):  # type: ignore
    def __init__(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> None:
        super().__init__(search_space)
        self._rng = np.random.RandomState()

        self.bounds = np.zeros((2, len(search_space)))

        for i, distribution in enumerate(search_space.values()):
            d = distribution
            assert isinstance(d, FloatDistribution)
            self.bounds[0, i] = d.low
            self.bounds[1, i] = d.high

        self.optimizer: PI_from_MaxSample | None = None

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        states = (optuna.trial.TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        if len(trials) < 1:
            return {}
        elif self.optimizer is None:
            X = np.zeros((len(trials), len(search_space)))
            for i, trial in enumerate(trials):
                X[i, :] = np.asarray(list(trial.params.values()))

            _sign = -1.0 if study.direction == optuna.study.StudyDirection.MINIMIZE else 1.0
            Y = np.zeros((len(trials), 1))
            for i, trial in enumerate(trials):
                Y[i, 0] = _sign * trial.value

            self.optimizer = PI_from_MaxSample(
                X=X,
                Y=Y,
                bounds=self.bounds,
            )
        else:
            assert self.optimizer is not None
            X_new = np.asarray(list(trials[-1].params.values()))
            _sign = -1.0 if study.direction == optuna.study.StudyDirection.MINIMIZE else 1.0
            Y_new = _sign * trials[-1].value

            if len(trials) % 5 == 4:
                self.optimizer.update(np.atleast_2d(X_new), np.atleast_2d(Y_new), optimize=True)
            else:
                self.optimizer.update(np.atleast_2d(X_new), np.atleast_2d(Y_new), optimize=False)

        new_inputs = self.optimizer.next_input()

        params = {}
        for name, value in zip(search_space.keys(), new_inputs[0]):
            params[name] = value
        return params
