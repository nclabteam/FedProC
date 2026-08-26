from collections import OrderedDict
from typing import Any


class Attack:
    """Base class for FL Byzantine attacks.

    Subclass and override craft() to implement an attack.
    The attack operates server-side: it receives all round packages after
    client training and before server aggregation, and may modify the
    packages of malicious clients.
    """

    def craft(
        self,
        packages: OrderedDict[int, dict[str, Any]],
        malicious_ids: list[int],
        ctx: Any,
    ) -> OrderedDict[int, dict[str, Any]]:
        """Modify malicious clients' packages before aggregation.

        Args:
            packages: all round packages keyed by client ID
            malicious_ids: subset of package keys that are Byzantine this round
            ctx: the sFL server instance (access to current_iter,
                 public_model_params, malicious_ids, configs, etc.)

        Returns:
            packages (modified in-place is fine; must return the dict)
        """
        return packages
