from click import BadParameter as BadParameter
from click import echo as echo
from typer.main import Typer as Typer

def Argument(
    default: str | None = ...,
    *,
    exists: bool = ...,
    dir_okay: bool = ...,
    readable: bool = ...,
) -> str: ...

def Option(default: str | None = ..., *param_decls: str) -> str: ...
