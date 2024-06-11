import random as rd
from emager_py.utils.majority_vote import MajorityVote

q = MajorityVote(10)

for _ in range(100):
    r = rd.randint(0, 6)
    q.append(r)
    print(r, q.vote())