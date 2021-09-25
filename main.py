
# 4장 구현

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



# 3. 게임개발
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
# # 8. 문자열 재정렬
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