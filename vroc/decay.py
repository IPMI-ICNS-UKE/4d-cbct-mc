from math import exp, log


def exponential_decay_factor(t: float, lambda_: float) -> float:
    return exp(-lambda_ * t)


def lambda_to_half_life(lambda_: float) -> float:
    return log(2) / lambda_


def half_life_to_lambda(half_life: float) -> float:
    return log(2) / half_life


def exponential_decay(
    initial_value: float,
    i_level: int,
    i_iteration: int,
    level_lambda: float,
    iteration_lambda: float,
) -> float:
    level_decay_factor = exponential_decay_factor(t=i_level, lambda_=level_lambda)
    itertion_decay_factor = exponential_decay_factor(
        t=i_iteration, lambda_=iteration_lambda
    )

    return initial_value * level_decay_factor * itertion_decay_factor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    initial_tau = 2.0
    vals = []

    for i_level in range(3):
        for i_iteration in range(100):
            tau = exponential_decay(
                initial_tau,
                i_level=i_level,
                i_iteration=i_iteration,
                level_lambda=0.46209812037329684,
                iteration_lambda=0.00,
            )
            vals.append(tau)

    plt.plot(vals)
