# 시간제한 2초, 메모리 제한 256 MB 
# N = 6 
# time complexity: O(36_C_3) = print(factorial(36) / (factorial(33) * factorial(3))) =  7140 
# f= open('Input.txt', 'r')
N = int(input())
graph = []
all_pos = []

for _ in range(N):
    graph.append(input().split())

DY = [-1, 1, 0,0]; DX = [0, 0, -1, 1]

# print(graph)
def factorial(num):
    if num <= 1:
        return num 
    return num * factorial(num-1)

def make_three_obstacles(level, seen):
    global all_pos 
    if level ==3:
        # print(seen)
        all_pos.append(list(seen))
        return 
    
    for y in range(N):
        for x in range(N):
            # 다른 obstacle과 위치가 겹치지 않고 학생이나 선생이 있지 않은 위치 
            if (y, x) not in seen and graph[y][x] == 'X':
                seen.add((y, x))
                make_three_obstacles(level+1, seen)
                seen.remove((y,x))

def find_teachers():
    teachers = []
    for y in range(N):
        for x in range(N):
            if graph[y][x] == 'T':
                teachers.append([y, x])
    return teachers 

def in_range(y, x):
    return 0 <= y < N and 0 <= x < N 

def can_avoid_teachers(teacher_poses, ob_poses):
    
    for ty, tx in teacher_poses:     
        for t in range(4):
            ny = ty + DY[t]; nx = tx + DX[t]  # 1방향으로 가기 전에 현재 선생 위치 초기화 
            while in_range(ny, nx):
                if graph[ny][nx] == 'S': # 학생을 먼저 만나면 
                    return False 
                # if graph[ny][nx] == 'O': # 장애물을 먼저 만나면 
                if (ny, nx) in ob_poses:
                    break # while break 후 다른 방향 찾기 
                ny = ny + DY[t]; nx = nx + DX[t]
    
    return True 

def solve():
    # 3개 장애물 선정 -> all_poses에 2차 배열로 저장 
    seen = set()
    make_three_obstacles(0, seen)
    # print(all_pos)

    # 선생 위치 파악 
    teachers = find_teachers()
    # print(teachers)

    # 선택된 장애물의 3개 위치에 대하여 
    # 감시를 피할 수 있는지 확인하고 하나라도 감시를 피할 수 있으면 바로 return 
    for poses in all_pos:
        # print(type(poses)) # list 
        if can_avoid_teachers(teachers, poses):
            print('YES') 
            return 
    
    print('NO')
    return 



if __name__ == '__main__':
    solve()

