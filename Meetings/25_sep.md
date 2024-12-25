# Preparation

- explain the number of blocks in a $TD(k,n)$
  - use the counting style $S = \{(p,l) | p \in l\}$
  - count from the perspective of a point $p$ and a line $l$
- explain the number of traversals in a $TD(k,n)$
- use sagemath to generate block graphs of $OA(m,n)$
  - after that try to switch the graph
  - after that try to compute the coherant rank

## Things to present

## Things found out

- the graph $K_{n,n}$ must be contained in the traversal of the blocks of the $TD(k,n)$, and only once
  - need to find a way to explain that it's only once, and the other $k-2$ choice of groups to form the block must be mutually exclusive of other blocks (since it would violate the definition of a TD which is that each pair of points across different groups must be in exactly one block)
  - start with $TD(2,n)$ and show that the number of blocks is $n^2$ due to the structure looking like $K_{n,n}$
  - build onto $TD(3,n)$ and show that the number of blocks is still $n^2$ 
