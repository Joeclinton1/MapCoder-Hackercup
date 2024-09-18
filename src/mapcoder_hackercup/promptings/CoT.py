from typing import List
import tiktoken
import os
from copy import deepcopy

from .Base import BaseStrategy
from mapcoder_hackercup.models.Base import BaseModel

from mapcoder_hackercup.results.Results import Results

from mapcoder_hackercup.datasets.Dataset import Dataset
from mapcoder_hackercup.datasets.HackercupDataset import HackercupDataset

class CoTStrategy(BaseStrategy):
    def run_single_pass(self, item: dict):
        prompt = """
Three little pigs from all over the world are meeting for a convention! Every minute, a triple of 3 new pigs arrives on the convention floor. After the n-th minute, the convention ends.

The big bad wolf has learned about this convention, and he has an attack plan. At some minute in the convention, he will arrive and eat exactly x pigs. Then he will get away.

The wolf wants Gregor to help him figure out the number of possible attack plans that involve eating exactly x pigs for various values of x (1 ≤ x ≤ 3n). Two attack plans are considered different, if they occur at different times or if the sets of little pigs to eat are different.

Note that all queries are independent, that is, the wolf does not eat the little pigs, he only makes plans!

Input

The first line of input contains two integers n and q (1 ≤ n ≤ 10^6, 1 ≤ q ≤ 2⋅ 10^5), the number of minutes the convention lasts and the number of queries the wolf asks.

Each of the next q lines contains a single integer x_i (1 ≤ x_i ≤ 3n), the number of pigs the wolf will eat in the i-th query.

Output

You should print q lines, with line i representing the number of attack plans if the wolf wants to eat x_i pigs. Since each query answer can be large, output each answer modulo 10^9+7.

Examples

Input


2 3
1
5
6


Output


9
6
1


Input


5 4
2
4
6
8


Output


225
2001
6014
6939

Note

In the example test, n=2. Thus, there are 3 pigs at minute 1, and 6 pigs at minute 2. There are three queries: x=1, x=5, and x=6.

If the wolf wants to eat 1 pig, he can do so in 3+6=9 possible attack plans, depending on whether he arrives at minute 1 or 2.

If the wolf wants to eat 5 pigs, the wolf cannot arrive at minute 1, since there aren't enough pigs at that time. Therefore, the wolf has to arrive at minute 2, and there are 6 possible attack plans.

If the wolf wants to eat 6 pigs, his only plan is to arrive at the end of the convention and devour everybody.

Remember to output your answers modulo 10^9+7!

## Let's think step by step and generate code to solve the problem.

```python
def inverses(n, P):
    inv = [1] * (n + 1)
    for i in range(2, n + 1):
        inv[i] = inv[P % i] * (P - P // i) % P
    return inv

mod = 10**9 + 7

def init():
    fac[0] = ifac[0] = 1
    inv = inverses(N, mod)
    for i in range(1, N):
        fac[i] = (fac[i - 1] * i) % mod
        ifac[i] = (ifac[i - 1] * inv[i]) % mod

def comb(n, k):
    return (fac[n] * ifac[n - k] * ifac[k]) % mod

N = 3 * 10**6 + 10

fac = [0] * N
ifac = [0] * N

init()

def main():
    n, Q = map(int, input().split())
    sz = 3 * (n + 1)

    # Create a list to store coefficients of polynomial p(x)
    p = [0] * (sz + 1)

    # Initialize coefficients of p(x) using binomial coefficients
    for i in range(1, sz + 1):
        p[i] = comb(sz, i)

    # Backward iteration to calculate coefficients of p(x-1) and p(x-2)
    for i in range(sz, 2, -1):
        p[i - 1] = ((p[i - 1] - 3 * p[i]) % mod + mod) % mod
        p[i - 2] = ((p[i - 2] - 3 * p[i]) % mod + mod) % mod

    # Remove extra elements at the beginning of p
    p = p[3:]

    # Answer queries based on calculated coefficients
    for _ in range(Q):
        x = int(input())
        print(p[x])

# Call the main function
main()
```

-------

This is the easy version of the problem. The only difference from the hard version is that in this version all coordinates are even.

There are n fence-posts at distinct coordinates on a plane. It is guaranteed that no three fence posts lie on the same line.

There are an infinite number of cows on the plane, one at every point with integer coordinates.

Gregor is a member of the Illuminati, and wants to build a triangular fence, connecting 3 distinct existing fence posts. A cow strictly inside the fence is said to be enclosed. If there are an odd number of enclosed cows and the area of the fence is an integer, the fence is said to be interesting.

Find the number of interesting fences.

Input

The first line contains the integer n (3 ≤ n ≤ 6000), the number of fence posts which Gregor can choose to form the vertices of a fence.

Each of the next n line contains two integers x and y (0 ≤ x,y ≤ 10^7, x and y are even), where (x,y) is the coordinate of a fence post. All fence posts lie at distinct coordinates. No three fence posts are on the same line.

Output

Print a single integer, the number of interesting fences. Two fences are considered different if they are constructed with a different set of three fence posts.

Examples

Input


3
0 0
2 0
0 4


Output


1


Input


5
0 0
2 16
30 14
4 6
2 10


Output


3

Note

In the first example, there is only 1 fence. That fence is interesting since its area is 4 and there is 1 enclosed cow, marked in red.

In the second example, there are 3 interesting fences. 

  * (0,0) — (30,14) — (2,10) 
  * (2,16) — (30,14) — (2,10) 
  * (30,14) — (4,6) — (2,10) 

## Let's think step by step and generate code to solve the problem.

```python
def main():
    # Read the number of fence posts
    n = int(input())

    # Array to store the count of fence posts based on parity of coordinates
    cnt = [[0, 0], [0, 0]]

    # Loop to read the coordinates of each fence post
    for _ in range(n):
        x, y = map(int, input().split())

        # Divide coordinates by 2 to handle even coordinates
        x >>= 1
        y >>= 1

        # Update the count based on the parity of coordinates
        cnt[x & 1][y & 1] += 1

    # Variable to store the number of interesting fences
    ans = 0

    # Loop to iterate over all combinations of parity for x and y coordinates
    for i in range(2):
        for j in range(2):
            k = cnt[i][j]

            # Calculate the number of interesting fences using combinations
            ans += 1 * k * (k - 1) // 2 * (k - 2) // 3 + 1 * k * (k - 1) // 2 * (n - k)

    # Print the result
    print(ans)

# Call the main function
main()
```

-------

Gregor is learning about RSA cryptography, and although he doesn't understand how RSA works, he is now fascinated with prime numbers and factoring them.

Gregor's favorite prime number is P. Gregor wants to find two bases of P. Formally, Gregor is looking for two integers a and b which satisfy both of the following properties.

  * P mod a = P mod b, where x mod y denotes the remainder when x is divided by y, and 
  * 2 ≤ a < b ≤ P. 



Help Gregor find two bases of his favorite prime number!

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 1000).

Each subsequent line contains the integer P (5 ≤ P ≤ {10}^9), with P guaranteed to be prime.

Output

Your output should consist of t lines. Each line should consist of two integers a and b (2 ≤ a < b ≤ P). If there are multiple possible solutions, print any.

Example

Input


2
17
5


Output


3 5
2 4

Note

The first query is P=17. a=3 and b=5 are valid bases in this case, because 17 mod 3 = 17 mod 5 = 2. There are other pairs which work as well.

In the second query, with P=5, the only solution is a=2 and b=4.

## Let's think step by step and generate code to solve the problem.

def main():
    # Read the number of test cases
    t = int(input())

    # List to store prime numbers for each test case
    primes = [int(input()) for _ in range(t)]

    # Iterate over each test case to find bases
    for prime in primes:
        # Print bases a and b for each test case
        print(f"2 {prime - 1}")

# Call the main function
main()

"""
        
        
        input_for_planning = [
            {
                "role": "user",
                "content": f"{prompt}\n{self.data.get_prompt(item)}\n## Let's think step by step and generate {self.language} code to solve the problem.\n# ----------------\nImportant: Your response must contain only the {self.language} code to solve this problem inside ``` block.",
            },
        ]
        return self.gpt_chat(
            processed_input=input_for_planning
        )

