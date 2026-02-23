
from typing import List 


class TreeAncestor:
    def __init__(self, n: int, parent: List[int]):
        self.n = n 
        self.LOG = (self.n).bit_length()
        self.up = [[-1]*n for _ in range(self.LOG)]
        self.up[0] = parent[:] # self.up[j][v] = node v를 2^j번 점프했을때 parent 

        # self.up INIT 
        for j in range(1, self.LOG): # 각 이진수 자리 j에 대해 
            for v in range(n): # v
                # 0~j-1까지 채워졌다고 가정하면, 
                # 2^j = 2^j-1 + 2^j-1 즉,v의 2^j-1의 parent의 2^j-1 parent가 됨. 
                if self.up[j-1][v] != -1:
                    self.up[j][v] = self.up[j-1][self.up[j-1][v]]
                else:
                    self.up[j][v] = -1 



    def getKthAncestor(self, node: int, k: int) -> int: 
        for j in reversed(range(self.LOG)):
            if node == -1: # 이미 루트를 넘었으면 더 볼 필요 없음
                break 
            if k & (1 << j): # k의 j번째 비트가 1인지 확인, 즉 "2^j" 점프가 필요한가?
                node = self.up[j][node] # 필요하다면, 한 번에 2^j 조상으로 이동 

        return node 