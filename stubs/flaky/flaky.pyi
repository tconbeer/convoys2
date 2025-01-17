from types import TracebackType
from typing import Callable, Protocol, TypeVar

from .flaky_pytest_plugin import FlakyPlugin

T_args_contra = TypeVar("T_args_contra", contravariant=True)
T_kwargs_contra = TypeVar("T_kwargs_contra", contravariant=True)
T_ret_co = TypeVar("T_ret_co", covariant=True)

class WrappedFunc(Protocol[T_args_contra, T_kwargs_contra, T_ret_co]):
    def __call__(self, *args: T_args_contra, **kwds: T_kwargs_contra) -> T_ret_co: ...

def flaky(
    max_runs: WrappedFunc | int | None = None,
    min_passes: int | None = None,
    rerun_filter: Callable[
        [
            tuple[type[BaseException], BaseException, TracebackType]
            | tuple[None, None, None],
            str,
            Callable,
            FlakyPlugin,
        ],
        bool,
    ]
    | None = None,
) -> WrappedFunc | Callable[[WrappedFunc], T_ret_co]: ...
