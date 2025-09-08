import subprocess, sys, time, difflib, pathlib

PYTHON = sys.executable            # 현재 파이썬
SOLUTION = "solution.py"           # 채점할 파일
TEST_DIR = pathlib.Path("tests")
IN_DIR = TEST_DIR / "in"
OUT_DIR = TEST_DIR / "out"
TIME_LIMIT_SEC = 2                 # 테스트케이스당 시간 제한(초)
NORMALIZE_TRAILING_SPACES = True   # 줄 끝 공백 무시

def read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")

def normalize(s: str) -> str:
    s = s.replace("\r\n", "\n")
    if NORMALIZE_TRAILING_SPACES:
        s = "\n".join(line.rstrip() for line in s.splitlines())
    return s.strip("\n") + "\n"

def run_case(in_path: pathlib.Path) -> tuple[bool, float, str]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [PYTHON, SOLUTION],
            stdin=open(in_path, "rb"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIME_LIMIT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - t0, f"TIMEOUT > {TIME_LIMIT_SEC}s"

    dt = time.perf_counter() - t0
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    return True, dt, out if proc.returncode == 0 else f"RUNTIME ERROR:\n{err}"

def main():
    cases = sorted(IN_DIR.glob("*.txt"))
    if not cases:
        print("No testcases in tests/in/*.txt")
        return

    total = len(cases); ok = 0
    for i, in_path in enumerate(cases, 1):
        out_path = OUT_DIR / in_path.name
        print(f"\n[{i}/{total}] {in_path.name}")
        if not out_path.exists():
            print(f"  ✖ expected file missing: {out_path}")
            continue

        ran, dt, got = run_case(in_path)
        if not ran:
            print(f"  ✖ did not run")
            continue

        exp = read_text(out_path)
        g_norm, e_norm = normalize(got), normalize(exp)
        if g_norm == e_norm:
            ok += 1
            print(f"  ✔ PASS  ({dt*1000:.1f} ms)")
        else:
            print(f"  ✖ FAIL  ({dt*1000:.1f} ms)")
            print("  --- diff (got vs expected) ---")
            diff = difflib.unified_diff(
                g_norm.splitlines(), e_norm.splitlines(),
                fromfile="got", tofile="expected", lineterm=""
            )
            for line in diff:
                print("  " + line)
            # 필요하면 표준에러도 확인
            # print("  stderr:\n", err)

    print(f"\nSummary: {ok}/{total} passed")

if __name__ == "__main__":
    main()
