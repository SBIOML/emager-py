from collections import deque
from scipy import stats


class MajorityVote(deque):
    def __init__(self, max_len):
        """
        MajorityVote object abstracts the classic majority voting algorithm.

        :param max_len: int, the maximum length of the queue

        Example:
        ```python
        import random as rd
        q = MajorityVote(10)
        for _ in range(100):
            r = rd.randint(0, 6)
            q.append(r)
            vote = q.vote()
        ```
        """
        super().__init__(maxlen=max_len)

    def vote(self):
        return stats.mode(self).mode


if __name__ == "__main__":
    import random as rd

    q = MajorityVote(10)

    for _ in range(100):
        r = rd.randint(0, 6)
        q.append(r)
        print(r, q.vote().mode)
