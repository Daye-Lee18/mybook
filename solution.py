class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        DP = [0]*len(s) # index 저장 
        new_start_idx = -1

        for idx, char in enumerate(s):
            # if new_start_idx == -1:
            #     return False 
            
            cur_idx = t[new_start_idx+1:].find(char)
    
            if cur_idx == -1:
                return False 

            DP[idx] = (new_start_idx + 1) + cur_idx # length of string to the previous char in the original  string + cur char index after previous char 
            new_start_idx = DP[idx]
        
        return True 
    

if __name__ == "__main__":
    s = 'abc'; t = 'ahbgdc'
    # s = 'a'; t = 'a'
    # s = 'axc'; t = 'ahbgdc'
    sol = Solution()
    print(sol.isSubsequence(s, t))