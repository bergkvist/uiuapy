# UiuaPy

## Installation
```sh
pip install uiuapy
```

If building the source distribution, rather than installing a precompiled wheel, you will need to have [rustup](https://rustup.rs/) installed on your machine.

## Usage
```py
import uiua

print(uiua.compile('/+')([1, 2, 3]))
# 6.0

print(uiua.compile('⌕')('ab', 'abracabra')) 
# [1 0 0 0 0 1 0 0 0]
```


## NumPy integration
UiuaPy uses the [NumPy C-API](https://numpy.org/doc/2.1/reference/c-api/index.html) for taking in Python inputs and returning Python results.

Uiua supports 5 data-types for arrays/scalars:

|Uiua type|NumPy equivalent dtype|
|---------|----------------------|
|Num|float64|
|Byte|uint8|
|Complex|complex128|
|Char|Unicode (32-bit characters)|
|Box|object|

If you pass in a NumPy array that does not have one of the above dtypes, it will be automatically converted according to the table below:
|NumPy dtype|Converted NumPy dtype|Uiua type|
|-|-|-|
|float32|float64|Num|
|uint64|float64|Num|
|uint32|float64|Num|
|uint16|float64|Num|
|int64|float64|Num|
|int32|float64|Num|
|int16|float64|Num|
|bool|uint8|Byte|

#### Interfacing overhead
Passing a numpy array to uiua requires copying its memory (using `memcpy`/ [`std::ptr::copy_nonoverlapping`](https://doc.rust-lang.org/beta/std/ptr/fn.copy_nonoverlapping.html) in Rust). The same is true for the values returned from the Uiua computation.

If using anything other than float64, uint8, complex128 or unicode data - there are also type conversion costs to take into account.

## Development

### Dependencies
- [rustup](https://rustup.rs/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

```sh
# Format code
uv run ruff format
cargo fmt

# Fix/report lints
uv run ruff check --fix
cargo clippy

# Run tests
uv run pytest

# Build wheels
uv build
```
