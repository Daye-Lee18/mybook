# p1.py — Robot Return to Origin

class Solution:
    def judgeCircle(self, moves: str) -> bool:
        x = y = 0
        for ch in moves:
            if ch == 'U':
                y += 1
            elif ch == 'D':
                y -= 1
            elif ch == 'L':
                x -= 1
            elif ch == 'R':
                x += 1
        return x == 0 and y == 0


def run_tests():
    sol = Solution()
    tests = [
        ("UD", True),
        ("LL", False),
        ("RULD", True),
        ("RRRRLLLL", True),
        ("UUDDLRLR", True),
        ("UDLR", True),
        ("UUUDDDRRLL", True),
        ("U", False),
        ("RRDLLU", True),
        ("URDL", True),                      # 10: 한 바퀴 순환
        ("UDUDUD", True),                    # 11: 왕복 반복
        ("LLLLRRRRUU", False),               # 12: 위로 2 남음
        ("UDLRU", False),                    # 13: 위로 1 남음
        ("UUDDRRLL", True),                  # 14: 축별로 상쇄
        ("L"*5 + "R"*3, False),              # 15: 왼쪽 2 남음
        ("U"*10 + "D"*10 + "L"*20 + "R"*20, True),  # 16: 대형 균형
        ("U"*100 + "D"*99, False),           # 17: 위로 1 남음
        ("DRUL", True),                      # 18: 순서만 다른 완전 상쇄
        ("RRLLUD", True),                   # 19: 아래로 1 남음
        ("R"*250 + "L"*250 + "U"*123 + "D"*123, True),  # 20: 큰 수 상쇄
    ]
    passed = 0
    for i, (moves, expected) in enumerate(tests, 1):
        got = sol.judgeCircle(moves)
        ok = got == expected
        print(f"[p1][case {i}] moves={moves!r} -> {got} (expected {expected}) {'OK' if ok else 'FAIL'}")
        passed += ok
    print(f"[p1] Passed {passed}/{len(tests)}")


if __name__ == "__main__":
    run_tests()
