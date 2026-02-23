# Coding Algorithm 

## 홈페이지 빌드 방법 

1) 로컬에서 실시간 (http)로 보기 
```bash
# (처음 1번) 설치
pip install -U jupyter-book

# 레포 폴더에서 빌드
jupyter-book build .

# 생성된 HTML을 로컬 서버로 띄우기
cd _build/html
python -m http.server 8000

# localhost:8000 으로 확인!
```

2) Github push 및 deploy 
3) 
```bash
git add .
git commit -m "update book"
git push origin main
```

1) 수동 배포 (gh-pages 브랜치에 HTML 올리는 방식)
```bash
# pip install -U jupyter-book ghp-import

jupyter-book build .
ghp-import -n -p -f _build/html
```