'''
이 문제는 괄호를 어디에 치느냐에 따라 결과가 달라진다.
즉, 앞에서부터 누적하는 문제가 아니라 어떤 구간 [i, j]를 어디서 나눌지(k)가 중요하므로 구간 DP로 푼다.

state:
dp[i][j] = nums[i] ~ nums[j] 구간을 계산해서 만들 수 있는
           (최솟값, 최댓값)

왜 min/max 둘 다 필요한가?
- '+' 는 min끼리, max끼리 보면 되지만
- '-' 는 최댓값을 만들 때 "왼쪽 최대 - 오른쪽 최소",
        최솟값을 만들 때 "왼쪽 최소 - 오른쪽 최대"가 필요하다.
따라서 각 구간의 min/max를 모두 저장해야 한다.

transition:
구간 [i, j]를 k에서 나누면
(i ~ k) op[k] (k+1 ~ j)

for k in range(i, j):
    left = dp[i][k]
    right = dp[k+1][j]
    op = ops[k]
'''

def solution(arr):
    def calculator(start, end):
        # [start, end] 구간에서 
        nonlocal dp, arr
       
        res_min, res_max = INF_MAX, INF_MIN
        for k in range(start, end): # XXX: [start, end-1]
            left = dp[start][k]
            right = dp[k+1][end]
            # NOTE: ops의 개수 = 숫자의 개수 -1 
            if ops[k] == "+":
                cur_max = left[1] + right[1]
                cur_min = left[0] + right[0]
            else: # 뺄셈 
                cur_max = left[1] - right[0] # 왼쪽 큰 거 - 오른쪽 작은거 
                cur_min = left[0] - right[1]# 왼쪽 작은 거 - 오른쪽 큰 거

            res_min = min(res_min, cur_min)
            res_max = max(res_max, cur_max)

        return [res_min, res_max]

    n = (len(arr) + 1 ) // 2 # 숫자 개수 

    nums = list(map(int, arr[::2]))
    ops = arr[1::2]

    # DP INIT [min, max]
    INF_MIN = -1 * int(1e9)
    INF_MAX = int(1e9)
    dp = [[[INF_MAX, INF_MIN] for _ in range(n)] for _ in range(n)] # NOTE: 숫자 개수만큼의 dp table 생성

    # base case: dp[i][i] = [arr[i], arr[i]]
    for i in range(n):
        dp[i][i] = [nums[i], nums[i]]

    # 길이 늘려가면서
    for length in range(2, n+1): # range: [2, n]
        for start in range(0, n-length+1): # range: [0, n-length]
            end = start + length -1  
            dp[start][end] = calculator(start, end) # dp[i][j] = 해당 구간 interval 

    return dp[0][n-1][1] # 숫자 구간 (0, n)에서의 최댓값 



if __name__ == "__main__":
    arr = ["1", "-", "3", "+", "5", "-", "8"] # result: 1 
    print(solution(arr))