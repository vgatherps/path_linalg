Linear algebra libraries I had written to support minimum-cost path findig with negative cycles

Computers are quite fast at linear algebra so doing this as opposed to fancy graph searches turns out to be faster for most of the matrix dimensions that are relevant to my use case

The dual minplus allow recovering the path alongside the cost by assigning a unique prime to each edge, and summing log(primes) using the same mask used to select the costs