from typing import Any, Callable, Dict, Optional

import wrapt


class CallbackWrapper(wrapt.ObjectProxy):
    """A transparent wrapper that updates its value via callback.

    Args:
        callback: Function that takes an environment dict and returns a new value
        default_value: Initial value to wrap (defaults to None)
        constant: If True, prevents updates (defaults to False)
    """

    def __init__(
        self,
        callback: Callable[[Dict[str, Any]], Any],
        default_value: Optional[Any] = None,
        constant: bool = False,
    ):
        # Initialize the ObjectProxy with the default_value
        super().__init__(default_value)

        # Store callback and constant flag in the wrapper
        self._self_callback = callback
        self._self_constant = constant

    def update_value(self, **env: Dict[str, Any]) -> None:
        """Update the wrapped value by calling the callback with the environment."""
        if not self._self_constant:
            # Update the wrapped object by setting self.__wrapped__
            self.__wrapped__ = self._self_callback(env)
