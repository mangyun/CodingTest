# 3장 그리디

# 1. 큰 수의 법칙
# 일일이 다 하는 것보단 원리를 생각해 수열을 나열하여 해결
# N, M, K를 공백을 기준으로 구분하여 입력 받기
# n, m, k = map(int, input().split())
# # N개의 수를 공백을 기준으로 구분하여 입력 받기
# data = list(map(int, input().split()))

# data.sort() # 입력 받은 수들 정렬하기
# first = data[n - 1] # 가장 큰 수
# second = data[n - 2] # 두 번째로 큰 수

# # 가장 큰 수가 더해지는 횟수 계산
# count = int(m / (k + 1)) * k
# count += m % (k + 1)

# result = 0
# result += (count) * first # 가장 큰 수 더하기
# result += (m - count) * second # 두 번째로 큰 수 더하기

# print(result) # 최종 답안 출력

# 2. 숫자 카드 게임
# N, M을 공백을 기준으로 구분하여 입력 받기
# n, m = map(int, input().split())

# result = 0
# # 한 줄씩 입력 받아 확인하기
# for i in range(n):
#     data = list(map(int, input().split()))
#     # 현재 줄에서 '가장 작은 수' 찾기
#     min_value = 10001
#     for a in data:
#         min_value = min(min_value, a)
#     # '가장 작은 수'들 중에서 가장 큰 수 찾기
#     result = max(result, min_value)

# print(result) # 최종 답안 출력

# 3. 1이 될 때까지
# # N, K공백을 기준으로 구분하여 입력 받기
# n, k = map(int, input().split())

# result = 0

# while True:
#     # N이 K로 나누어 떨어지는 수가 될 때까지만 1씩 빼기
#     target = (n // k) * k
#     result += (n - target)
#     n = target
#     # N이 K보다 작을 때 (더 이상 나눌 수 없을 때) 반복문 탈출
#     if n < k:
#         break
#     # K로 나누기
#     result += 1
#     n //= k

# # 마지막으로 남은 수에 대하여 1씩 빼기
# result += (n - 1)
# print(result)














# 4장 구현
# 구현 문제는 조건들을 전부 만들어야함.
# 사소한 입력 조건등을 문제에서 설명해주며, 문제 길이가 긴편임. 고차원적인 편은 아니라, 문법만 숙지하면 쉽게 풀 수 있음.
# 코테에서 가장 난이도가 낮은 1, 2번은 구현과 그리디 문제이다.

# 4-1. 상하좌우
# N 입력받기
# n = int(input())
# x, y = 1, 1
# plans = input().split()

# # L, R, U, D에 따른 이동 방향
# dx = [0, 0, -1, 1]
# dy = [-1, 1, 0, 0]
# move_types = ['L', 'R', 'U', 'D']

# # 이동 계획을 하나씩 확인
# for plan in plans:
#     # 이동 후 좌표 구하기
#     for i in range(len(move_types)):
#         if plan == move_types[i]:
#             nx = x + dx[i]
#             ny = y + dy[i]
#     # 공간을 벗어나는 경우 무시
#     if nx < 1 or ny < 1 or nx > n or ny > n:
#         continue
#     # 이동 수행
#     x, y = nx, ny

# print(x, y)


# 4-2. 시각
#이런 모든 경우의 수를 확인해보는 것은 완전 탐색이다. 100만개 이하일 때, 적절하다
# H를 입력받기
# h = int(input())

# count = 0
# for i in range(h + 1):
#     for j in range(60):
#         for k in range(60):
#             # 매 시각 안에 '3'이 포함되어 있다면 카운트 증가
#             if '3' in str(i) + str(j) + str(k):
#                 count += 1

# print(count)



# 2. 왕실의 나이트
# 구현과 그리디를 섞은 문제이며, 방향을 steps이나 dx, dy를 따로 지정해 범위안으로 계산.
# 현재 나이트의 위치 입력받기
# input_data = input()
# row = int(input_data[1])
# column = int(ord(input_data[0])) - int(ord('a')) + 1

# # 나이트가 이동할 수 있는 8가지 방향 정의
# steps = [(-2, -1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1)]

# # 8가지 방향에 대하여 각 위치로 이동이 가능한지 확인
# result = 0
# for step in steps:
#     # 이동하고자 하는 위치 확인
#     next_row = row + step[0]
#     next_column = column + step[1]
#     # 해당 위치로 이동이 가능하다면 카운트 증가
#     if next_row >= 1 and next_row <= 8 and next_column >= 1 and next_column <= 8:
#         result += 1

# print(result)




# 3. 게임개발 - 칸을 옮길때마다, 다시 방향전환 변수 리셋
# 방향문제에서는 dx, dy같이 방향을 먼저 설정해주기.
# # N, M을 공백을 기준으로 구분하여 입력받기
# n, m = map(int, input().split())

# # 방문한 위치를 저장하기 위한 맵을 생성하여 0으로 초기화
# d = [[0] * m for _ in range(n)]
# # 현재 캐릭터의 X 좌표, Y 좌표, 방향을 입력받기
# x, y, direction = map(int, input().split())
# d[x][y] = 1 # 현재 좌표 방문 처리

# # 전체 맵 정보를 입력받기
# array = []
# for i in range(n):
#     array.append(list(map(int, input().split())))

# # 북, 동, 남, 서 방향 정의
# dx = [-1, 0, 1, 0]
# dy = [0, 1, 0, -1]

# # 왼쪽으로 회전
# def turn_left():
#     global direction
#     direction -= 1
#     if direction == -1:
#         direction = 3

# # 시뮬레이션 시작
# count = 1
# turn_time = 0
# while True:
#     # 왼쪽으로 회전
#     turn_left()
#     nx = x + dx[direction]
#     ny = y + dy[direction]
#     # 회전한 이후 정면에 가보지 않은 칸이 존재하는 경우 이동
#     if d[nx][ny] == 0 and array[nx][ny] == 0:
#         d[nx][ny] = 1
#         x = nx
#         y = ny
#         count += 1
#         turn_time = 0
#         continue
#     # 회전한 이후 정면에 가보지 않은 칸이 없거나 바다인 경우
#     else:
#         turn_time += 1
#     # 네 방향 모두 갈 수 없는 경우
#     if turn_time == 4:
#         nx = x - dx[direction]
#         ny = y - dy[direction]
#         # 뒤로 갈 수 있다면 이동하기
#         if array[nx][ny] == 0:
#             x = nx
#             y = ny
#         # 뒤가 바다로 막혀있는 경우
#         else:
#             break
#         turn_time = 0
# # 정답 출력
# print(count)



# # 12장 구현 문제
# 7. 럭키 스트레이트
# #input()은 애초에 리스트 형태, 계산할때만 int()해주면 됨. 총 더한 수에서, 값빼기
# n = input()
# length = len(n) # 점수 값의 총 자릿수
# summary = 0

# # 왼쪽 부분의 자릿수의 합 더하기
# for i in range(length // 2):
#     summary += int(n[i])

# # 오른쪽 부분의 자릿수의 합 빼기
# for i in range(length // 2, length):
#     summary -= int(n[i])

# # 왼쪽 부분과 오른쪽 부분의 자릿수 합이 동일한지 검사
# if summary == 0:
#     print("LUCKY")
# else:
#     print("READY")


# # 8. 문자열 재정렬
# 알파벳인경우 isalpha()로 확인, 오름차순 sort(), 리스트를 문자열로 ''.join
# data = input()
# result = []
# value = 0

# # 문자를 하나씩 확인하며
# for x in data:
#     # 알파벳인 경우 결과 리스트에 삽입
#     if x.isalpha():
#         result.append(x)
#     # 숫자는 따로 더하기
#     else:
#         value += int(x)

# # 알파벳을 오름차순으로 정렬
# result.sort()

# # 숫자가 하나라도 존재하는 경우 가장 뒤에 삽입
# if value != 0:
#     result.append(str(value))

# # 최종 결과 출력(리스트를 문자열로 변환하여 출력)
# print(''.join(result))


# 9. 문자열 압축
# 완전 탐색이라 모든 경우를 다함.
# 여기서는 정규표현 import re를 해서 이용, s[0:step]처럼도 이용가능
# 가장 중요한 범위는 단어의 반을 넘어가게되면 의미가 없기때문에, len(s)//2를 해줌.
# import re
# s = input()
# def solution(s):

#   if len(s) <= 2:
#     return len(s)

#   resultCount = []
#   for i in range(1, len(s)//2+1):
#     reList = re.sub('(\w{%i})' %i, '\g<1> ', s).split()
#     count = 1
#     result = []
    
#     for j in range(len(reList)):
#       if j < len(reList) -1 and reList[j] == reList[j+1]:
#         count +=1
#       else:
#         if count ==1:
#           result.append(reList[j])
#         else:
#           result.append(str(count) + reList[j])
#           count=1

#     resultCount.append(len(''.join(result)))

#   return min(resultCount)
# print(solution(s))


# 10. 자물쇠와 열쇠
# 2차원 리스트 90도 회전하기
# 2차원 리스트를 확장시킴.
# 회전함수는 return list(zip(*a[::-1]))으로도 표현가능

# def rotate_a_matrix_by_90_degree(a):
#     n = len(a) # 행 길이 계산
#     m = len(a[0]) # 열 길이 계산
#     result = [[0] * n for _ in range(m)] # 결과 리스트
#     for i in range(n):
#         for j in range(m):
#             result[j][n - i - 1] = a[i][j]
#     return result

# # 자물쇠의 중간 부분이 모두 1인지 확인
# def check(new_lock):
#     lock_length = len(new_lock) // 3
#     for i in range(lock_length, lock_length * 2):
#         for j in range(lock_length, lock_length * 2):
#             if new_lock[i][j] != 1:
#                 return False
#     return True

# def solution(key, lock):
#     n = len(lock)
#     m = len(key)
#     # 자물쇠의 크기를 기존의 3배로 변환
#     new_lock = [[0] * (n * 3) for _ in range(n * 3)]
#     # 새로운 자물쇠의 중앙 부분에 기존의 자물쇠 넣기
#     for i in range(n):
#         for j in range(n):
#             new_lock[i + n][j + n] = lock[i][j]

#     # 4가지 방향에 대해서 확인
#     for rotation in range(4):
#         key = rotate_a_matrix_by_90_degree(key) # 열쇠 회전
#         for x in range(n * 2):
#             for y in range(n * 2):
#                 # 자물쇠에 열쇠를 끼워 넣기
#                 for i in range(m):
#                     for j in range(m):
#                         new_lock[x + i][y + j] += key[i][j]
#                 # 새로운 자물쇠에 열쇠가 정확히 들어 맞는지 검사
#                 if check(new_lock) == True:
#                     return True
#                 # 자물쇠에서 열쇠를 다시 빼기
#                 for i in range(m):
#                     for j in range(m):
#                         new_lock[x + i][y + j] -= key[i][j]
#     return False


# # 11. 뱀
# n = int(input())
# k = int(input())
# data = [[0] * (n + 1) for _ in range(n + 1)] # 맵 정보
# info = [] # 방향 회전 정보

# # 맵 정보(사과 있는 곳은 1로 표시)
# for _ in range(k):
#     a, b = map(int, input().split())
#     data[a][b] = 1

# # 방향 회전 정보 입력
# l = int(input())
# for _ in range(l):
#     x, c = input().split()
#     info.append((int(x), c))

# # 처음에는 오른쪽을 보고 있으므로(동, 남, 서, 북)
# dx = [0, 1, 0, -1]
# dy = [1, 0, -1, 0]

# def turn(direction, c):
#     if c == "L":
#         direction = (direction - 1) % 4
#     else:
#         direction = (direction + 1) % 4
#     return direction

# def simulate():
#     x, y = 1, 1 # 뱀의 머리 위치
#     data[x][y] = 2 # 뱀이 존재하는 위치는 2로 표시
#     direction = 0 # 처음에는 동쪽을 보고 있음
#     time = 0 # 시작한 뒤에 지난 '초' 시간
#     index = 0 # 다음에 회전할 정보
#     q = [(x, y)] # 뱀이 차지하고 있는 위치 정보(꼬리가 앞쪽)

#     while True:
#         nx = x + dx[direction]
#         ny = y + dy[direction]
#         # 맵 범위 안에 있고, 뱀의 몸통이 없는 위치라면
#         if 1 <= nx and nx <= n and 1 <= ny and ny <= n and data[nx][ny] != 2:
#             # 사과가 없다면 이동 후에 꼬리 제거
#             if data[nx][ny] == 0:
#                 data[nx][ny] = 2
#                 q.append((nx, ny))
#                 px, py = q.pop(0)
#                 data[px][py] = 0
#             # 사과가 있다면 이동 후에 꼬리 그대로 두기
#             if data[nx][ny] == 1:
#                 data[nx][ny] = 2
#                 q.append((nx, ny))
#         # 벽이나 뱀의 몸통과 부딪혔다면
#         else:
#             time += 1
#             break
#         x, y = nx, ny # 다음 위치로 머리를 이동
#         time += 1
#         if index < l and time == info[index][0]: # 회전할 시간인 경우 회전
#             direction = turn(direction, info[index][1])
#             index += 1
#     return time

# print(simulate())

# 12. 기둥과 보 설치
# in 이라는 리스트 안의 내용을 참조
# 먼저 설치 후, 조건에 맞게 삭제

# 현재 설치된 구조물이 '가능한' 구조물인지 확인하는 함수
# def possible(answer):
#     for x, y, stuff in answer:
#         if stuff == 0: # 설치된 것이 '기둥'인 경우
#             # '바닥 위' 혹은 '보의 한쪽 끝 부분 위' 혹은 '다른 기둥 위'라면 정상
#             if y == 0 or [x - 1, y, 1] in answer or [x, y, 1] in answer or [x, y - 1, 0] in answer:
#                 continue
#             return False # 아니라면 거짓(False) 반환
#         elif stuff == 1: # 설치된 것이 '보'인 경우
#             # '한쪽 끝부분이 기둥 위' 혹은 '양쪽 끝부분이 다른 보와 동시에 연결'이라면 정상
#             if [x, y - 1, 0] in answer or [x + 1, y - 1, 0] in answer or ([x - 1, y, 1] in answer and [x + 1, y, 1] in answer):
#                 continue
#             return False # 아니라면 거짓(False) 반환
#     return True

# def solution(n, build_frame):
#     answer = []
#     for frame in build_frame: # 작업(frame)의 개수는 최대 1,000개
#         x, y, stuff, operate = frame
#         if operate == 0: # 삭제하는 경우
#             answer.remove([x, y, stuff]) # 일단 삭제를 해본 뒤에
#             if not possible(answer): # 가능한 구조물인지 확인
#                 answer.append([x, y, stuff]) # 가능한 구조물이 아니라면 다시 설치
#         if operate == 1: # 설치하는 경우
#             answer.append([x, y, stuff]) # 일단 설치를 해본 뒤에
#             if not possible(answer): # 가능한 구조물인지 확인
#                 answer.remove([x, y, stuff]) # 가능한 구조물이 아니라면 다시 제거
#     return sorted(answer) # 정렬된 결과를 반환



#13. 치킨배달
# 이 문제는 고르는 문제이기 때문에, 순열과 조합에서 조합을 이용한다. 순열은 from itertools import permutations
# 순열은 순서대로 나열하기 때문에 '12'와 '21'은 다른 케이스로 봄, 하지만 조합은 뽑기만 하기 때문에 '12'와 '21'은 같은 케이스로 구분하게 되어 '12'만 결과에 나오게 됨

# from itertools import combinations

# n, m = map(int, input().split())
# chicken, house = [], []

# for r in range(n):
#     data = list(map(int, input().split()))
#     for c in range(n):
#         if data[c] == 1:
#             house.append((r, c)) # 일반 집
#         elif data[c] == 2:
#             chicken.append((r, c)) # 치킨집

# # 모든 치킨 집 중에서 m개의 치킨 집을 뽑는 조합 계산
# candidates = list(combinations(chicken, m))

# # 치킨 거리의 합을 계산하는 함수
# def get_sum(candidate):
#     result = 0
#     # 모든 집에 대하여
#     for hx, hy in house:
#         # 가장 가까운 치킨 집을 찾기
#         temp = 1e9
#         for cx, cy in candidate:
#             temp = min(temp, abs(hx - cx) + abs(hy - cy))
#         # 가장 가까운 치킨 집까지의 거리를 더하기
#         result += temp
#     # 치킨 거리의 합 반환
#     return result

# # 치킨 거리의 합의 최소를 찾아 출력
# result = 1e9
# for candidate in candidates:
#     result = min(result, get_sum(candidate))

# print(result)


# 14. 외벽점검
# from itertools import permutations

# def solution(n, weak, dist):
#     # 길이를 2배로 늘려서 '원형'을 일자 형태로 변형
#     length = len(weak)
#     for i in range(length):
#         weak.append(weak[i] + n)
#     answer = len(dist) + 1 # 투입할 친구 수의 최솟값을 찾아야 하므로 len(dist) + 1로 초기화
#     # 0부터 length - 1까지의 위치를 각각 시작점으로 설정
#     for start in range(length):
#         # 친구를 나열하는 모든 경우 각각에 대하여 확인
#         for friends in list(permutations(dist, len(dist))):
#             count = 1 # 투입할 친구의 수
#             # 해당 친구가 점검할 수 있는 마지막 위치
#             position = weak[start] + friends[count - 1]
#             # 시작점부터 모든 취약한 지점을 확인
#             for index in range(start, start + length):
#                 # 점검할 수 있는 위치를 벗어나는 경우
#                 if position < weak[index]:
#                     count += 1 # 새로운 친구를 투입
#                     if count > len(dist): # 더 투입이 불가능하다면 종료
#                         break
#                     position = weak[index] + friends[count - 1]
#             answer = min(answer, count) # 최솟값 계산
#     if answer > len(dist):
#         return -1
#     return answer


# 5장 DFS/BFS
# DFS 특징 : 재귀 함수, 스택, 
# 3. 음료수 얼려먹기
# # N, M을 공백을 기준으로 구분하여 입력 받기
# n, m = map(int, input().split())

# # 2차원 리스트의 맵 정보 입력 받기
# graph = []
# for i in range(n):
#     graph.append(list(map(int, input())))

# # DFS로 특정한 노드를 방문한 뒤에 연결된 모든 노드들도 방문
# def dfs(x, y):
#     # 주어진 범위를 벗어나는 경우에는 즉시 종료
#     if x <= -1 or x >= n or y <= -1 or y >= m:
#         return False
#     # 현재 노드를 아직 방문하지 않았다면
#     if graph[x][y] == 0:
#         # 해당 노드 방문 처리
#         graph[x][y] = 1
#         # 상, 하, 좌, 우의 위치들도 모두 재귀적으로 호출
#         dfs(x - 1, y)
#         dfs(x, y - 1)
#         dfs(x + 1, y)
#         dfs(x, y + 1)
#         return True
#     return False

# # 모든 노드(위치)에 대하여 음료수 채우기
# result = 0
# for i in range(n):
#     for j in range(m):
#         # 현재 위치에서 DFS 수행
#         if dfs(i, j) == True:
#             result += 1

# print(result) # 정답 출력



# BFS 특징 : 최단거리, 큐 이용, 상하좌우 방향벡터 먼저 설정
# 4. 미로탈출
# from collections import deque

# # N, M을 공백을 기준으로 구분하여 입력 받기
# n, m = map(int, input().split())
# # 2차원 리스트의 맵 정보 입력 받기
# graph = []
# for i in range(n):
#     graph.append(list(map(int, input())))

# # 이동할 네 가지 방향 정의 (상, 하, 좌, 우)
# dx = [-1, 1, 0, 0]
# dy = [0, 0, -1, 1]

# # BFS 소스코드 구현
# def bfs(x, y):
#     # 큐(Queue) 구현을 위해 deque 라이브러리 사용
#     queue = deque()
#     queue.append((x, y))
#     # 큐가 빌 때까지 반복하기
#     while queue:
#         x, y = queue.popleft()
#         # 현재 위치에서 4가지 방향으로의 위치 확인
#         for i in range(4):
#             nx = x + dx[i]
#             ny = y + dy[i]
#             # 미로 찾기 공간을 벗어난 경우 무시
#             if nx < 0 or nx >= n or ny < 0 or ny >= m:
#                 continue
#             # 벽인 경우 무시
#             if graph[nx][ny] == 0:
#                 continue
#             # 해당 노드를 처음 방문하는 경우에만 최단 거리 기록
#             if graph[nx][ny] == 1:
#                 graph[nx][ny] = graph[x][y] + 1
#                 queue.append((nx, ny))
#     # 가장 오른쪽 아래까지의 최단 거리 반환
#     return graph[n - 1][m - 1]

# # BFS를 수행한 결과 출력
# print(bfs(0, 0))


# 13장 DFS/BFS 기출문제
# 15. 특정 거리의 도시 찾기
# x 에서 출발해서 작은거부터 확인
# # 최단거리가 k라면 리턴
# from collections import deque

# # 도시의 개수, 도로의 개수, 거리 정보, 출발 도시 번호
# n, m, k, x = map(int, input().split())
# graph = [[] for _ in range(n + 1)]

# # 모든 도로 정보 입력 받기
# for _ in range(m):
#     a, b = map(int, input().split())
#     graph[a].append(b)

# # 모든 도시에 대한 최단 거리 초기화
# distance = [-1] * (n + 1)
# distance[x] = 0 # 출발 도시까지의 거리는 0으로 설정

# # 너비 우선 탐색(BFS) 수행
# q = deque([x])
# while q:
#     now = q.popleft()
#     # 현재 도시에서 이동할 수 있는 모든 도시를 확인
#     for next_node in graph[now]:
#         # 아직 방문하지 않은 도시라면
#         if distance[next_node] == -1:
#             # 최단 거리 갱신
#             distance[next_node] = distance[now] + 1
#             q.append(next_node)

# # 최단 거리가 K인 모든 도시의 번호를 오름차순으로 출력
# check = False
# for i in range(1, n + 1):
#     if distance[i] == k:
#         print(i)
#         check = True

# # 만약 최단 거리가 K인 도시가 없다면, -1 출력
# if check == False:
#     print(-1)

16. 연구소


# 6장 정렬

# 4. 두 배열의 원소 교체
# n, k = map(int, input().split()) # N과 K를 입력 받기
# a = list(map(int, input().split())) # 배열 A의 모든 원소를 입력받기
# b = list(map(int, input().split())) # 배열 B의 모든 원소를 입력받기

# a.sort() # 배열 A는 오름차순 정렬 수행
# b.sort(reverse=True) # 배열 B는 내림차순 정렬 수행

# # 첫 번째 인덱스부터 확인하며, 두 배열의 원소를 최대 K번 비교
# for i in range(k):
#     # A의 원소가 B의 원소보다 작은 경우
#     if a[i] < b[i]:
#         # 두 원소를 교체
#         a[i], b[i] = b[i], a[i]
#     else: # A의 원소가 B의 원소보다 크거나 같을 때, 반복문을 탈출
#         break

# print(sum(a)) # 배열 A의 모든 원소의 합을 출력










# 7장 이진탐색
# 2. 떡볶이 떡 만들기

# # 떡의 개수(N)와 요청한 떡의 길이(M)을 입력
# n, m = list(map(int, input().split(' ')))
# # 각 떡의 개별 높이 정보를 입력
# array = list(map(int, input().split()))

# # 이진 탐색을 위한 시작점과 끝점 설정
# start = 0
# end = max(array)

# # 이진 탐색 수행 (반복적)
# result = 0
# while(start <= end):
#     total = 0
#     mid = (start + end) // 2
#     for x in array:
#         # 잘랐을 때의 떡볶이 양 계산
#         if x > mid:
#             total += x - mid
#     # 떡볶이 양이 부족한 경우 더 많이 자르기 (오른쪽 부분 탐색)
#     if total < m:
#         end = mid - 1
#     # 떡볶이 양이 충분한 경우 덜 자르기 (왼쪽 부분 탐색)
#     else:
#         result = mid # 최대한 덜 잘랐을 때가 정답이므로, 여기에서 result에 기록
#         start = mid + 1

# # 정답 출력
# print(result)

# 15장 이진 탐색 문제
# 1. 정렬된 배열에서 특정 수의 개수 구하기
# from bisect import bisect_left, bisect_right

# # 값이 [left_value, right_value]인 데이터의 개수를 반환하는 함수
# def count_by_range(array, left_value, right_value):
#     right_index = bisect_right(array, right_value)
#     left_index = bisect_left(array, left_value)
#     return right_index - left_index

# n, x = map(int, input().split()) # 데이터의 개수 N, 찾고자 하는 값 x 입력 받기
# array = list(map(int, input().split())) # 전체 데이터 입력 받기

# # 값이 [x, x] 범위에 있는 데이터의 개수 계산
# count = count_by_range(array, x, x)

# # 값이 x인 원소가 존재하지 않는다면
# if count == 0:
#     print(-1)
# # 값이 x인 원소가 존재한다면
# else:
#     print(count)











# 8장 다이나믹 프로그래밍
# 재귀형식, 구하고 싶은 값은 이전 리스트값들의 min, max를 더함
# 다이나믹 프로그래밍의 전형적인 형태는 보텀업 = 점화식 만들기
# 그리디, 구현, 완전탐색으로 시간복잡도가 낮아지지않는다면, 
# 다이나믹 프로그래밍 = 보텀업 = 상향식 = 반복문 = 리스트갯수만큼 초기화
# 메모이제이션 = 하향식 = 탑다운 = 캐싱

# 1. 1로 만들기
# # 정수 X를 입력 받기
# x = int(input())

# # 앞서 계산된 결과를 저장하기 위한 DP 테이블 초기화
# d = [0] * 1000001

# # 다이나믹 프로그래밍(Dynamic Programming) 진행(보텀업)
# for i in range(2, x + 1):
#     # 현재의 수에서 1을 빼는 경우
#     d[i] = d[i - 1] + 1
#     # 현재의 수가 2로 나누어 떨어지는 경우
#     if i % 2 == 0:
#         d[i] = min(d[i], d[i // 2] + 1)
#     # 현재의 수가 3으로 나누어 떨어지는 경우
#     if i % 3 == 0:
#         d[i] = min(d[i], d[i // 3] + 1)
#     # 현재의 수가 5로 나누어 떨어지는 경우
#     if i % 5 == 0:
#         d[i] = min(d[i], d[i // 5] + 1)

# print(d[x])


# 2. 개미 전사
# # 정수 N을 입력 받기
# n = int(input())
# # 모든 식량 정보 입력 받기
# array = list(map(int, input().split()))

# # 앞서 계산된 결과를 저장하기 위한 DP 테이블 초기화
# d = [0] * 100

# # 다이나믹 프로그래밍(Dynamic Programming) 진행 (보텀업)
# d[0] = array[0]
# d[1] = max(array[0], array[1]) 
# for i in range(2, n):
#     d[i] = max(d[i - 1], d[i - 2] + array[i])

# # 계산된 결과 출력
# print(d[n - 1])


# 4. 효율적인 화폐 구성
# # 정수 N, M을 입력 받기
# n, m = map(int, input().split())
# # N개의 화폐 단위 정보를 입력 받기
# array = []
# for i in range(n):
#     array.append(int(input()))

# # 한 번 계산된 결과를 저장하기 위한 DP 테이블 초기화
# d = [10001] * (m + 1)

# # 다이나믹 프로그래밍(Dynamic Programming) 진행(보텀업)
# d[0] = 0
# for i in range(n):
#     for j in range(array[i], m + 1):
#         if d[j - array[i]] != 10001: # (i - k)원을 만드는 방법이 존재하는 경우
#             d[j] = min(d[j], d[j - array[i]] + 1)

# # 계산된 결과 출력
# if d[m] == 10001: # 최종적으로 M원을 만드는 방법이 없는 경우
#     print(-1)
# else:
#     print(d[m])


# 16장 다이나믹 프로그래밍 문제
# 1. 금광
# # 테스트 케이스(Test Case) 입력
# for tc in range(int(input())):
#     # 금광 정보 입력
#     n, m = map(int, input().split())
#     array = list(map(int, input().split()))

#     # 다이나믹 프로그래밍을 위한 2차원 DP 테이블 초기화
#     dp = []
#     index = 0
#     for i in range(n):
#         dp.append(array[index:index + m])
#         index += m

#     # 다이나믹 프로그래밍 진행
#     for j in range(1, m):
#         for i in range(n):
#             # 왼쪽 위에서 오는 경우
#             if i == 0:
#                 left_up = 0
#             else:
#                 left_up = dp[i - 1][j - 1]
#             # 왼쪽 아래에서 오는 경우
#             if i == n - 1:
#                 left_down = 0
#             else:
#                 left_down = dp[i + 1][j - 1]
#             # 왼쪽에서 오는 경우
#             left = dp[i][j - 1]
#             dp[i][j] = dp[i][j] + max(left_up, left_down, left)

#     result = 0
#     for i in range(n):
#         result = max(result, dp[i][m - 1])

#     print(result)


# 4. 병사 배치하기
# n = int(input())
# array = list(map(int, input().split()))
# # 순서를 뒤집어 '최장 증가 부분 수열' 문제로 변환
# array.reverse()

# # 다이나믹 프로그래밍을 위한 1차원 DP 테이블 초기화
# dp = [1] * n

# # 가장 긴 증가하는 부분 수열(LIS) 알고리즘 수행
# for i in range(1, n):
#     for j in range(0, i):
#         if array[j] < array[i]:
#             dp[i] = max(dp[i], dp[j] + 1)

# # 열외해야 하는 병사의 최소 수를 출력
# print(n - max(dp))









# 9장. 최단 경로 알고리즘

# import sys
# input = sys.stdin.readline
# INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# # 노드의 개수, 간선의 개수를 입력받기
# n, m = map(int, input().split())
# # 시작 노드 번호를 입력받기
# start = int(input())
# # 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
# graph = [[] for i in range(n + 1)]
# # 방문한 적이 있는지 체크하는 목적의 리스트를 만들기
# visited = [False] * (n + 1)
# # 최단 거리 테이블을 모두 무한으로 초기화
# distance = [INF] * (n + 1)

# # 모든 간선 정보를 입력받기
# for _ in range(m):
#     a, b, c = map(int, input().split())
#     # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
#     graph[a].append((b, c))

# # 방문하지 않은 노드 중에서, 가장 최단 거리가 짧은 노드의 번호를 반환
# def get_smallest_node():
#     min_value = INF
#     index = 0 # 가장 최단 거리가 짧은 노드(인덱스)
#     for i in range(1, n + 1):
#         if distance[i] < min_value and not visited[i]:
#             min_value = distance[i]
#             index = i
#     return index

# def dijkstra(start):
#     # 시작 노드에 대해서 초기화
#     distance[start] = 0
#     visited[start] = True
#     for j in graph[start]:
#         distance[j[0]] = j[1]
#     # 시작 노드를 제외한 전체 n - 1개의 노드에 대해 반복
#     for i in range(n - 1):
#         # 현재 최단 거리가 가장 짧은 노드를 꺼내서, 방문 처리
#         now = get_smallest_node()
#         visited[now] = True
#         # 현재 노드와 연결된 다른 노드를 확인
#         for j in graph[now]:
#             cost = distance[now] + j[1]
#             # 현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
#             if cost < distance[j[0]]:
#                 distance[j[0]] = cost

# # 다익스트라 알고리즘을 수행
# dijkstra(start)

# # 모든 노드로 가기 위한 최단 거리를 출력
# for i in range(1, n + 1):
#     # 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
#     if distance[i] == INF:
#         print("INFINITY")
#     # 도달할 수 있는 경우 거리를 출력
#     else:
#         print(distance[i])

# 파이썬은 1초에 약 2천만번의 연산이 가능하기 때문에 노드의 개수가 5천개 정도는 연산이 가능할 것이지만, 그 이상은 힘들것이다. 따라서 전형적인 순차탐색 다익스트라가 아닌, 아래의 우선순위 큐인 힙을 이용해 개선된 다익스트라를 실행해야한다.
# import heapq
# import sys
# input = sys.stdin.readline
# INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# # 노드의 개수, 간선의 개수를 입력받기
# n, m = map(int, input().split())
# # 시작 노드 번호를 입력받기
# start = int(input())
# # 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
# graph = [[] for i in range(n + 1)]
# # 최단 거리 테이블을 모두 무한으로 초기화
# distance = [INF] * (n + 1)

# # 모든 간선 정보를 입력받기
# for _ in range(m):
#     a, b, c = map(int, input().split())
#     # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
#     graph[a].append((b, c))

# def dijkstra(start):
#     q = []
#     # 시작 노드로 가기 위한 최단 경로는 0으로 설정하여, 큐에 삽입
#     heapq.heappush(q, (0, start))
#     distance[start] = 0
#     while q: # 큐가 비어있지 않다면
#         # 가장 최단 거리가 짧은 노드에 대한 정보 꺼내기
#         dist, now = heapq.heappop(q)
#         # 현재 노드가 이미 처리된 적이 있는 노드라면 무시
#         if distance[now] < dist:
#             continue
#         # 현재 노드와 연결된 다른 인접한 노드들을 확인
#         for i in graph[now]:
#             cost = dist + i[1]
#             # 현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 짧은 경우
#             if cost < distance[i[0]]:
#                 distance[i[0]] = cost
#                 heapq.heappush(q, (cost, i[0]))

# # 다익스트라 알고리즘을 수행
# dijkstra(start)

# # 모든 노드로 가기 위한 최단 거리를 출력
# for i in range(1, n + 1):
#     # 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
#     if distance[i] == INF:
#         print("INFINITY")
#     # 도달할 수 있는 경우 거리를 출력
#     else:
#         print(distance[i])




# 이제까지는 어느 한 노드에서 모든 노드를 찾는 다익스트라 과정이었다면, 지금부터는 모든노드에서 모든노드를 찾는 플로이드워셜 알고리즘이다. 노드의 갯수가 적은 경우 사용할 수 있다. 500개 이상 잘 안나옴.
# INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# # 노드의 개수 및 간선의 개수를 입력받기
# n = int(input())
# m = int(input())
# # 2차원 리스트(그래프 표현)를 만들고, 모든 값을 무한으로 초기화
# graph = [[INF] * (n + 1) for _ in range(n + 1)]

# # 자기 자신에서 자기 자신으로 가는 비용은 0으로 초기화
# for a in range(1, n + 1):
#     for b in range(1, n + 1):
#         if a == b:
#             graph[a][b] = 0

# # 각 간선에 대한 정보를 입력 받아, 그 값으로 초기화
# for _ in range(m):
#     # A에서 B로 가는 비용은 C라고 설정
#     a, b, c = map(int, input().split())
#     graph[a][b] = c

# # 점화식에 따라 플로이드 워셜 알고리즘을 수행
# for k in range(1, n + 1):
#     for a in range(1, n + 1):
#         for b in range(1, n + 1):
#             graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

# # 수행된 결과를 출력
# for a in range(1, n + 1):
#     for b in range(1, n + 1):
#         # 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
#         if graph[a][b] == 1e9:
#             print("INFINITY", end=" ")
#         # 도달할 수 있는 경우 거리를 출력
#         else:
#             print(graph[a][b], end=" ")
#     print()




# 1. 전보 - 다익스트라 힙
# import heapq
# import sys
# input = sys.stdin.readline
# INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# # 노드의 개수, 간선의 개수, 시작 노드를 입력받기
# n, m, start = map(int, input().split())
# # 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
# graph = [[] for i in range(n + 1)]
# # 최단 거리 테이블을 모두 무한으로 초기화
# distance = [INF] * (n + 1)

# # 모든 간선 정보를 입력받기
# for _ in range(m):
#     x, y, z = map(int, input().split())
#     # X번 노드에서 Y번 노드로 가는 비용이 Z라는 의미
#     graph[x].append((y, z))

# def dijkstra(start):
#    q = []
#    # 시작 노드로 가기 위한 최단 경로는 0으로 설정하여, 큐에 삽입
#    heapq.heappush(q, (0, start))
#    distance[start] = 0
#    while q: # 큐가 비어있지 않다면
#         # 가장 최단 거리가 짧은 노드에 대한 정보를 꺼내기
#         dist, now = heapq.heappop(q)
#         if distance[now] < dist:
#             continue
#         # 현재 노드와 연결된 다른 인접한 노드들을 확인
#         for i in graph[now]:
#             cost = dist + i[1]
#             # 현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 짧은 경우
#             if cost < distance[i[0]]:
#                 distance[i[0]] = cost
#                 heapq.heappush(q, (cost, i[0]))

# # 다익스트라 알고리즘을 수행
# dijkstra(start)

# # 도달할 수 있는 노드의 개수
# count = 0
# # 도달할 수 있는 노드 중에서, 가장 멀리 있는 노드와의 최단 거리
# max_distance = 0
# for d in distance:
#     # 도달할 수 있는 노드인 경우
#     if d != 1e9:
#         count += 1
#         max_distance = max(max_distance, d)

# # 시작 노드는 제외해야 하므로 count - 1을 출력
# print(count - 1, max_distance)


# 2. 미래 도시 - 플로이드 워셜
# INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# # 노드의 개수 및 간선의 개수를 입력받기
# n, m = map(int, input().split())
# # 2차원 리스트(그래프 표현)를 만들고, 모든 값을 무한으로 초기화
# graph = [[INF] * (n + 1) for _ in range(n + 1)]

# # 자기 자신에서 자기 자신으로 가는 비용은 0으로 초기화
# for a in range(1, n + 1):
#     for b in range(1, n + 1):
#         if a == b:
#             graph[a][b] = 0

# # 각 간선에 대한 정보를 입력 받아, 그 값으로 초기화
# for _ in range(m):
#     # A와 B가 서로에게 가는 비용은 1이라고 설정
#     a, b = map(int, input().split())
#     graph[a][b] = 1
#     graph[b][a] = 1

# # 거쳐 갈 노드 X와 최종 목적지 노드 K를 입력받기
# x, k = map(int, input().split())

# # 점화식에 따라 플로이드 워셜 알고리즘을 수행
# for k in range(1, n + 1):
#     for a in range(1, n + 1):
#         for b in range(1, n + 1):
#             graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

# # 수행된 결과를 출력
# distance = graph[1][k] + graph[k][x]

# # 도달할 수 없는 경우, -1을 출력
# if distance >= 1e9:
#     print("-1")
# # 도달할 수 있다면, 최단 거리를 출력
# else:
#     print(distance)










# 10장. 기타 그래프 이론
# 개선된 서로소 집합 알고리즘 - 경로 압축
# # 특정 원소가 속한 집합을 찾기
# def find_parent(parent, x):
#     # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
#     if parent[x] != x:
#         parent[x] = find_parent(parent, parent[x])
#     return parent[x]

# # 두 원소가 속한 집합을 합치기
# def union_parent(parent, a, b):
#     a = find_parent(parent, a)
#     b = find_parent(parent, b)
#     if a < b:
#         parent[b] = a
#     else:
#         parent[a] = b

# # 노드의 개수와 간선(Union 연산)의 개수 입력 받기
# v, e = map(int, input().split())
# parent = [0] * (v + 1) # 부모 테이블 초기화하기

# # 부모 테이블상에서, 부모를 자기 자신으로 초기화
# for i in range(1, v + 1):
#     parent[i] = i

# # Union 연산을 각각 수행
# for i in range(e):
#     a, b = map(int, input().split())
#     union_parent(parent, a, b)

# # 각 원소가 속한 집합 출력하기
# print('각 원소가 속한 집합: ', end='')
# for i in range(1, v + 1):
#     print(find_parent(parent, i), end=' ')

# print()

# # 부모 테이블 내용 출력하기
# print('부모 테이블: ', end='')
# for i in range(1, v + 1):
#     print(parent[i], end=' ')


# 서로소 집합을 활용한 사이클 판별
# # 특정 원소가 속한 집합을 찾기
# def find_parent(parent, x):
#     # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
#     if parent[x] != x:
#         parent[x] = find_parent(parent, parent[x])
#     return parent[x]

# # 두 원소가 속한 집합을 합치기
# def union_parent(parent, a, b):
#     a = find_parent(parent, a)
#     b = find_parent(parent, b)
#     if a < b:
#         parent[b] = a
#     else:
#         parent[a] = b

# # 노드의 개수와 간선(Union 연산)의 개수 입력 받기
# v, e = map(int, input().split())
# parent = [0] * (v + 1) # 부모 테이블 초기화하기

# # 부모 테이블상에서, 부모를 자기 자신으로 초기화
# for i in range(1, v + 1):
#     parent[i] = i

# cycle = False # 사이클 발생 여부

# for i in range(e):
#     a, b = map(int, input().split())
#     # 사이클이 발생한 경우 종료
#     if find_parent(parent, a) == find_parent(parent, b):
#         cycle = True
#         break
#     # 사이클이 발생하지 않았다면 합집합(Union) 연산 수행
#     else:
#         union_parent(parent, a, b)

# if cycle:
#     print("사이클이 발생했습니다.")
# else:
#     print("사이클이 발생하지 않았습니다.")


# 크루스칼 알고리즘 - 모든 선인 연결된 사이클이 없는 신장트리
# # 특정 원소가 속한 집합을 찾기
# def find_parent(parent, x):
#     # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
#     if parent[x] != x:
#         parent[x] = find_parent(parent, parent[x])
#     return parent[x]

# # 두 원소가 속한 집합을 합치기
# def union_parent(parent, a, b):
#     a = find_parent(parent, a)
#     b = find_parent(parent, b)
#     if a < b:
#         parent[b] = a
#     else:
#         parent[a] = b

# # 노드의 개수와 간선(Union 연산)의 개수 입력 받기
# v, e = map(int, input().split())
# parent = [0] * (v + 1) # 부모 테이블 초기화하기

# # 모든 간선을 담을 리스트와, 최종 비용을 담을 변수
# edges = []
# result = 0

# # 부모 테이블상에서, 부모를 자기 자신으로 초기화
# for i in range(1, v + 1):
#     parent[i] = i

# # 모든 간선에 대한 정보를 입력 받기
# for _ in range(e):
#     a, b, cost = map(int, input().split())
#     # 비용순으로 정렬하기 위해서 튜플의 첫 번째 원소를 비용으로 설정
#     edges.append((cost, a, b))

# # 간선을 비용순으로 정렬
# edges.sort()

# # 간선을 하나씩 확인하며
# for edge in edges:
#     cost, a, b = edge
#     # 사이클이 발생하지 않는 경우에만 집합에 포함
#     if find_parent(parent, a) != find_parent(parent, b):
#         union_parent(parent, a, b)
#         result += cost

# print(result)



# 위상정렬 - 싸이클이 없는 방향 그래프, 큐의 나간 순서 그대로 출력
# from collections import deque

# # 노드의 개수와 간선의 개수를 입력 받기
# v, e = map(int, input().split())
# # 모든 노드에 대한 진입차수는 0으로 초기화
# indegree = [0] * (v + 1)
# # 각 노드에 연결된 간선 정보를 담기 위한 연결 리스트 초기화
# graph = [[] for i in range(v + 1)]

# # 방향 그래프의 모든 간선 정보를 입력 받기
# for _ in range(e):
#     a, b = map(int, input().split())
#     graph[a].append(b) # 정점 A에서 B로 이동 가능
#     # 진입 차수를 1 증가
#     indegree[b] += 1

# # 위상 정렬 함수
# def topology_sort():
#     result = [] # 알고리즘 수행 결과를 담을 리스트
#     q = deque() # 큐 기능을 위한 deque 라이브러리 사용

#     # 처음 시작할 때는 진입차수가 0인 노드를 큐에 삽입
#     for i in range(1, v + 1):
#         if indegree[i] == 0:
#             q.append(i)

#     # 큐가 빌 때까지 반복
#     while q:
#         # 큐에서 원소 꺼내기
#         now = q.popleft()
#         result.append(now)
#         # 해당 원소와 연결된 노드들의 진입차수에서 1 빼기
#         for i in graph[now]:
#             indegree[i] -= 1
#             # 새롭게 진입차수가 0이 되는 노드를 큐에 삽입
#             if indegree[i] == 0:
#                 q.append(i)

#     # 위상 정렬을 수행한 결과 출력
#     for i in result:
#         print(i, end=' ')

# topology_sort()