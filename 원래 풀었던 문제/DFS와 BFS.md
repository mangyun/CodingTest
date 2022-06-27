### 문제

https://www.acmicpc.net/problem/1260

<br>


### 코드

```python
#dfs 부분
def dfs(v):
  visited_list[v] = 1
  print(v, end=' ')
  for i in range(n+1):
    if(visited_list[i] == 0 and matrix[v][i] == 1):
      dfs(i)

#bfs 부분
from collections import deque

def bfs(v):
  q = deque()
  q.append(v)
  visited_list[v] = 1

  while q: #q가 빌때까지
    v = q.popleft()
    print(v, end=' ')
    for i in range(1, n+1):
      if visited_list[i] == 0 and matrix[v][i] == 1:
        q.append(i)
        visited_list[i] = 1

#정점, 간선 입력
n, m, v = map(int, input().split())
matrix = [[0]*(n+1) for i in range(n+1)]
visited_list = [0]*(n+1)

for i in range(m):
  a, b = map(int, input().split())
  matrix[a][b] = matrix[b][a] = 1
       
dfs(v)
visited_list = [0]*(n+1)  
print()
bfs(v)
```


<br>

### 느낀점
dfs는 달리기 바통터치같은 느낌, 스택과 재귀함수를 이용한다.
bfs는 와이파이 공유기같은 느낌, 큐와 덱으로 구현한다.

<br>

### 기억하기!
```python
  def bfs(v):
    q = deque()
    q.append(v)
    visited_list[v] = 1

    while q: #q가 빌때까지
      v = q.popleft()
      print(v, end=' ')
      for i in range(1, n+1):
        if visited_list[i] == 0 and matrix[v][i] == 1:
          q.append(i)
          visited_list[i] = 1
```
bfs는 시작점을 큐에 한번 추가하고, 한번씩 빼가면서 그때마다 추가해준다는것을 기억

<br>

```python
  def dfs(v):
    visited_list[v] = 1
    print(v, end=' ')
    for i in range(n+1):
      if(visited_list[i] == 0 and matrix[v][i] == 1):
        dfs(i)
```
dfs는 한번에 쭉 재귀호출 한다는것을 기억




