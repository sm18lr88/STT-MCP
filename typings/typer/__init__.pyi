from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

class Typer:
    def __init__(self, *, no_args_is_help: bool = ...) -> None: ...
    def __call__(self) -> None: ...
    def command(self, name: str | None = ...) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...

def Argument(
    default: str | None = ...,
    *,
    exists: bool = ...,
    dir_okay: bool = ...,
    readable: bool = ...,
) -> str: ...

def Option(default: str | None = ..., *param_decls: str) -> str: ...
