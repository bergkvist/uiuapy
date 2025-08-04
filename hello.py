import uiua
import numpy as np

def main():
    sum = uiua.compile('/+')
    print(f'{sum(np.array([1, 2, 3])) = }')

if __name__ == "__main__":
    main()
