# Preparation

- Explain the formula between OA(m,n) and SRG(n, k, λ, μ), specifically how many common neighbours two non-adjacent vertices have.
- Read up on coherant ranks and how they relate to the clique graph of the SRG.

## Things to do

- Make slides for the presentation
  - Answer: $μ=m(m-1)$
  - Explanation:
    - we have m rows to choose the first match between $v_i$ and $v_{k}$, and since $v_i$ and $v_j$ cannot be adjacent, we must choose the second match from the remaining $m-1$ rows for $v_k$ and $v_j$ to be adjacent.

## After meeting

- Proof was not strong enough, had to account that after choosing the rows, the elements in the rows $r_1$ and $r_2$ must have only 1 choice -> due to OA property or ordered pairs, and subsequently every row is also fixed, meaning only 1 $v_k$ of that chosen $r_1$ and $r_2$

## Next steps

- read wikipedia page for the matrix construction of OA
- read Douglas R Stinson's book on combinatorial designs, 6.4 (MacNeish’s Theorem), 6.5, 6.7 (Wilson's Construction)
- use sagemath to play with the N = n/2 shit and set m = (n-3)/2
