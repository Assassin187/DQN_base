import random
import numpy as np

class MazeEnv:
    def __init__(self,size):
        self.size = size
        self.actions = [0,1,2,3]
        self.maze, self.start, self.end = self.generate(size)
        self.state = np.expand_dims(self.maze,axis=2).copy()
 
    def reset(self,complete=False):
        if complete:
            # 重置迷宫
            self.maze, self.start, self.end = self.generate(self.size)
        self.state = np.expand_dims(self.maze,axis=2)
        self.position = self.start
        self.goal = self.end
        self.path = [self.start]
        return self.state
 
    def step(self, action):
        # 执行动作
        next_position = None
        if action == 0 and self.position[0] > 0:
            next_position = (self.position[0]-1, self.position[1])
        elif action == 1 and self.position[0] < self.size-1:
            next_position = (self.position[0]+1, self.position[1])
        elif action == 2 and self.position[1] > 0:
            next_position = (self.position[0], self.position[1]-1)
        elif action == 3 and self.position[1] < self.size-1:
            next_position = (self.position[0], self.position[1]+1)
        else:
            next_position = self.position
 
        if next_position == self.goal:
            reward = 500
        elif self.maze[next_position] == -1:
            reward = -300
        else:
            reward = -10
 
        self.position = next_position  # 更新位置
        self.path.append(self.position)  # 加入路径
 
        next_state = self.state.copy()
        next_state[self.position] = 2 # 标记路径
 
        done = (self.position == self.goal)  # 判断是否结束
        return next_state, reward, done
 
    @staticmethod
    # 生成迷宫图像
    def generate(size):
        maze = np.zeros((size, size))
        # Start and end points
        start = (random.randint(0, size-1), 0)
        end = (random.randint(0, size-1), size-1)
        maze[start] = 1
        maze[end] = 1
        # Generate maze walls
        for i in range(size * size):
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            if (x, y) == start or (x, y) == end:
                continue
            if random.random() < 0.2:
                maze[x, y] = -1
            if np.sum(np.abs(maze)) == size*size - 2:
                break
        return maze, start, end
 
    @staticmethod
    # BFS求出路径
    def solve_maze(maze, start, end):
        size = maze.shape[0]
        visited = np.zeros((size, size))
        solve = np.zeros((size,size))
        queue = [start]
        visited[start[0],start[1]] = 1
        while queue:
            x, y = queue.pop(0)
            if (x, y) == end:
                break
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= size or ny < 0 or ny >= size or visited[nx, ny] or maze[nx, ny] == -1:
                    continue
                queue.append((nx, ny))
                visited[nx, ny] = visited[x, y] + 1
        if visited[end[0],end[1]] == 0:
            return solve,[]
        path = [end]
        x, y = end
        while (x, y) != start:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= size or ny < 0 or ny >= size or visited[nx, ny] != visited[x, y] - 1:
                    continue
                path.append((nx, ny))
                x, y = nx, ny
                break
        points = path[::-1]  # 倒序
        for point in points:
            solve[point[0]][point[1]] = 1
        return solve, points