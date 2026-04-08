import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ssl_pipeline import run_pseudo_labeling_ssl


@dataclass
class Particle:
    """
    A particle for PSO-based hyperparameter optimization.
    position = [threshold, max_pseudo_labels_per_round, pseudo_weight]
    velocity = same dimension as position
    """
    position: List[float]
    velocity: List[float]
    best_position: List[float]
    best_fitness: float


class PSOOptimizer:
    def __init__(
        self,
        swarm_size: int = 5,
        max_iters: int = 5,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int = 42,
        ssl_kwargs: Optional[Dict] = None,
        verbose: bool = True
    ):
        self.swarm_size = swarm_size
        self.max_iters = max_iters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        self.verbose = verbose

        random.seed(seed)
        np.random.seed(seed)

        self.ssl_kwargs = {
            "data_dir": "./data",
            "batch_size": 64,
            "val_ratio": 0.1,
            "labeled_ratio": 0.1,
            "seed": seed,
            "num_workers": 2,
            "ssl_rounds": 2,
            "epochs_per_round": 3,
            "learning_rate": 0.001,
            "save_model": False,
            "verbose": False
        }
        if ssl_kwargs is not None:
            self.ssl_kwargs.update(ssl_kwargs)

        # Search bounds
        self.threshold_bounds = (0.90, 0.99)
        self.max_pseudo_bounds = (200, 1500)
        self.pseudo_weight_bounds = (0.2, 1.0)

        self.swarm: List[Particle] = []
        self.global_best_position = None
        self.global_best_fitness = float("-inf")

        self.history = []

    def initialize_particle(self) -> Particle:
        threshold = random.uniform(*self.threshold_bounds)
        max_pseudo = random.uniform(*self.max_pseudo_bounds)
        pseudo_weight = random.uniform(*self.pseudo_weight_bounds)

        position = [threshold, max_pseudo, pseudo_weight]

        velocity = [
            random.uniform(-0.02, 0.02),   # threshold velocity
            random.uniform(-100, 100),     # pseudo-count velocity
            random.uniform(-0.1, 0.1)      # pseudo-weight velocity
        ]

        return Particle(
            position=position,
            velocity=velocity,
            best_position=position.copy(),
            best_fitness=float("-inf")
        )

    def clip_position(self, position: List[float]) -> List[float]:
        threshold = min(max(position[0], self.threshold_bounds[0]), self.threshold_bounds[1])
        max_pseudo = min(max(position[1], self.max_pseudo_bounds[0]), self.max_pseudo_bounds[1])
        pseudo_weight = min(max(position[2], self.pseudo_weight_bounds[0]), self.pseudo_weight_bounds[1])
        return [threshold, max_pseudo, pseudo_weight]

    def decode_position(self, position: List[float]) -> Tuple[float, int, float]:
        threshold = float(position[0])
        max_pseudo = int(round(position[1]))
        pseudo_weight = float(position[2])
        return threshold, max_pseudo, pseudo_weight

    def evaluate_fitness(self, position: List[float]) -> float:
        threshold, max_pseudo, pseudo_weight = self.decode_position(position)

        if self.verbose:
            print("\nEvaluating particle with:")
            print(f"  threshold     = {threshold:.4f}")
            print(f"  max_pseudo    = {max_pseudo}")
            print(f"  pseudo_weight = {pseudo_weight:.4f}")

        ssl_run_kwargs = dict(self.ssl_kwargs)
        ssl_run_kwargs.update({
            "threshold": threshold,
            "max_pseudo_labels_per_round": max_pseudo,
            "pseudo_weight": pseudo_weight
        })

        results = run_pseudo_labeling_ssl(**ssl_run_kwargs)

        fitness = results["best_val_acc"]
        if self.verbose:
            print(f"  fitness (val acc) = {fitness:.4f}")

        return fitness

    def initialize_swarm(self):
        if self.verbose:
            print("Initializing swarm...")
        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = float("-inf")
        self.history = []

        for i in range(self.swarm_size):
            particle = self.initialize_particle()
            fitness = self.evaluate_fitness(particle.position)

            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()

            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()

            self.swarm.append(particle)

            if self.verbose:
                print(f"Initialized particle {i+1}/{self.swarm_size}")

        self.history.append(self.global_best_fitness)

    def optimize(self):
        self.initialize_swarm()

        for iteration in range(1, self.max_iters + 1):
            if self.verbose:
                print(f"\n========== PSO Iteration {iteration}/{self.max_iters} ==========")

            for i, particle in enumerate(self.swarm):
                new_velocity = []
                new_position = []

                for d in range(3):
                    r1 = random.random()
                    r2 = random.random()

                    cognitive = self.c1 * r1 * (particle.best_position[d] - particle.position[d])
                    social = self.c2 * r2 * (self.global_best_position[d] - particle.position[d])

                    v_new = self.w * particle.velocity[d] + cognitive + social
                    x_new = particle.position[d] + v_new

                    new_velocity.append(v_new)
                    new_position.append(x_new)

                particle.velocity = new_velocity
                particle.position = self.clip_position(new_position)

                fitness = self.evaluate_fitness(particle.position)

                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

                if self.verbose:
                    print(
                        f"Particle {i+1}: "
                        f"best_fitness={particle.best_fitness:.4f}, "
                        f"global_best={self.global_best_fitness:.4f}"
                    )

            self.history.append(self.global_best_fitness)

        best_threshold, best_max_pseudo, best_pseudo_weight = self.decode_position(self.global_best_position)

        return {
            "best_threshold": best_threshold,
            "best_max_pseudo": best_max_pseudo,
            "best_pseudo_weight": best_pseudo_weight,
            "best_fitness": self.global_best_fitness,
            "history": self.history
        }


if __name__ == "__main__":
    optimizer = PSOOptimizer(
        swarm_size=3,
        max_iters=2,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=42
    )

    results = optimizer.optimize()

    print("\n======= FINAL PSO RESULTS =======")
    print(f"Best threshold     : {results['best_threshold']:.4f}")
    print(f"Best max pseudo    : {results['best_max_pseudo']}")
    print(f"Best pseudo weight : {results['best_pseudo_weight']:.4f}")
    print(f"Best validation acc: {results['best_fitness']:.4f}")
    print(f"Fitness history    : {results['history']}")
