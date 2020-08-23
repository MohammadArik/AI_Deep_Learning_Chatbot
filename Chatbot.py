import numpy as np

numbers = np.arange(1, 10, 1)
prime_numbers = [2]
for pnum in prime_numbers:
    for num in numbers:
        if num % pnum != 0:
            prime_numbers.append(num)


print(prime_numbers)

