import torch
import numpy as np
from typing import Optional
from torch import Tensor
from cmaes import CMA


class CMAESWrapper:
    def __init__(
        self,
        device: torch.device,
        dimension: int,
        population_size: int,
        mean: Tensor,
        sigma: float,
        minimize: bool = True,
        bounds: Optional[Tensor] = None,
        dtype_out: torch.dtype = torch.float32,
    ):
        if not isinstance(dimension, int):
            raise TypeError("dimension argument must be an int for CMAWrapper")
        if not isinstance(population_size, int):
            raise TypeError("population_size argument must be an int for CMAWrapper")
        if not isinstance(sigma, float):
            raise TypeError("sigma argument must be a float for CMAWrapper")
        if not isinstance(device, torch.device):
            raise TypeError("device argument must be a torch.device for CMAWrapper")

        self.dimension = dimension
        self.population_size = population_size
        self.mean = mean
        self.sigma = sigma
        self.minimize = minimize
        self.bounds = bounds
        self.device = device
        self.dtype_out = dtype_out

        self.cma = CMA(
            mean=mean.cpu().numpy(),
            sigma=sigma,
            bounds=bounds.cpu().numpy() if bounds is not None else None,
            population_size=population_size,
        )

        if self.minimize:
            self.best_fitness = torch.inf
        else:
            self.best_fitness = -torch.inf

        self.first_told = False

    def ask(self):
        candidates_np = np.array([self.cma.ask() for _ in range(self.population_size)])
        return torch.from_numpy(candidates_np).to(self.device).to(self.dtype_out)

    def first_tell(self, candidate, fitness):
        if not candidate.shape == (1, self.dimension):
            raise ValueError(
                f"Start solution of shape {candidate.shape} is not shape ({1}, {self.dimension}) for CMAWrapper"
            )

        if self.first_told:
            raise RuntimeError("first_tell has already been called for this CMAWrapper")

        self.first_told = True

        self.best_fitness = fitness.item()
        self.best_solution = candidate.cpu().numpy()[0]

    def tell(self, candidates, fitnesses):
        if (
            not candidates.shape == (self.population_size, self.dimension)
            and candidates.shape[0] != 1
        ):
            raise ValueError(
                f"solutions argument of shape {candidates.shape} is not shape ({self.population_size}, {self.dimension}) for CMAWrapper"
            )
        if not fitnesses.shape == (self.population_size,):
            raise ValueError(
                f"fitnesses argument of shape {fitnesses.shape} is not shape ({self.population_size},) for CMAWrapper"
            )

        # cma tell expects a list of tuples of (np.array, float)
        candidates = candidates.cpu().numpy()
        fitnesses = fitnesses.cpu().numpy()

        # update best solution CMA minimizes, so we need to negate the fitnesses if we are maximizing
        best_batch_index = (
            np.argmin(fitnesses) if self.minimize else np.argmax(fitnesses)
        )
        best_batch_fitness = fitnesses[best_batch_index]

        if self.minimize:
            tell_list = list(zip(candidates, fitnesses))
            if best_batch_fitness < self.best_fitness:
                self.best_fitness = best_batch_fitness
                self.best_solution = candidates[best_batch_index]
        else:
            tell_list = list(zip(candidates, -1.0 * fitnesses))
            if best_batch_fitness > self.best_fitness:
                self.best_fitness = best_batch_fitness
                self.best_solution = candidates[best_batch_index]

        self.cma.tell(tell_list)

    def should_stop(self):
        return self.cma.should_stop()

    def get_best_solution(self):
        return torch.from_numpy(self.best_solution).to(self.device)[None, :]

    def get_best_fitness(self):
        return self.best_fitness
