import logging

logger = logging.getLogger("mace_fep")


class LambdaSchedule:
    def __init__(self, start: float, delta: float, n_steps: int, use_ssc: bool):
        self.start = start
        self.delta = delta
        self.n_steps = n_steps
        self.current_lambda = start
        self.output_lambda = start
        self.transform = self.ssc_lambda if use_ssc else self.linear

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_steps

    def ssc_lambda(self, lmbda: float) -> float:
        return 6 * lmbda**5 - 15 * lmbda**4 + 10 * lmbda**3

    def linear(self, lmbda: float) -> float:
        return lmbda

    def __next__(self):
        output = self.transform(self.current_lambda)
        self.current_lambda += self.delta
        self.output_lambda = output
