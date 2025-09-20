from collections import deque 

N, T = map(int, input().split())
beverages = [] # T, C, M, TC, TM, CM, TCM 종류 
faiths = []
for n in range(1, N+1):
    beverages.append(list(input()))

for n in range(1, N+1):
    faiths.append(map(int, input().split()))


DY = [-1, 1, 0, 0]
DX = [0, 0, -1, 1]

def in_the_morning():
    for y in range(N):
        for x in range(N):
            faiths[y][x] += 1 

def group(y, x):
    if parents[(y, x)] == 0:
        parents[(y, x)] = (y, x) # 자기자신이 부모 
    
    
def in_the_lunch_time():
    # 인접한 학생들과 신봉 음식이 "완전히" 같은 경우에만 "그룹" 형성 
    
    # 그룹 내 대표자 한 명 선정: 신앙심이 가장 큼 -> 행이 작음 -> 열이 작음 

    # 대표자 제외 그룹원들은 각자 신앙심을 1씩 대표자에게 넘김 

def solve():
    # T일 동안, 하루는 아침, 점심, 저녁의 순서로 아래와 같은 과정 반복 
    # 아침: 신앙심 + 1 
    in_the_morning()

    # 점심: 
    in_the_lunch_time()

