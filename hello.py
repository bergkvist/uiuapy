import uiua
import numpy as np

def main():
    program = uiua.compile('/+')
    xs = np.linspace(0, 1, 100_000)
    result = program(xs)
    print(f'{result = }')

if __name__ == "__main__":
    main()
