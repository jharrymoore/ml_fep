import logging

logger = logging.getLogger("mace_fep")


class LambdaSchedule:
    def __init__(
        self,
        last_step: float,
        total_steps: int,
        delta: float,
        reverse: bool,
        n_steps: int,
        use_ssc: bool,
    ):
        self.last_step = last_step
        self.delta = delta
        self.reverse = reverse
        self.n_steps = n_steps
        # fraction through the schedule
        self.current_lambda = (
            self.last_step / total_steps
            if not reverse
            else 1 - self.last_step / total_steps
        )
        self.transform = self.ssc_lambda if use_ssc else self.linear
        self.output_lambda = self.transform(self.current_lambda)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_steps

    def ssc_lambda(self, lmbda: float) -> float:
        return 6 * lmbda**5 - 15 * lmbda**4 + 10 * lmbda**3

    def linear(self, lmbda: float) -> float:
        return lmbda

    def __next__(self):
        self.current_lambda += self.delta
        self.output_lambda = self.transform(self.current_lambda)
