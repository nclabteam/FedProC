from .base import Attack


class NoAttack(Attack):
    """No-op: benign mode. Returns packages unmodified."""
