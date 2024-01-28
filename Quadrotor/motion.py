from .quadrotor import quadrotor


class motion:

    def __init__(self, quadrotor: quadrotor) -> None:
        self.quadrotor = quadrotor
