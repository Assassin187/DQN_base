from matplotlib import pyplot as plt
import numpy as np
from maze import MazeEnv
from dqn import DQNAgent
from PIL import Image

def maze_to_image(maze, path):
    size = maze.shape[0]
    img = Image.new('RGB', (size, size), (255, 255, 255))
    pixels = img.load()
    for i in range(size):
        for j in range(size):
            if maze[i, j] == -1:
                pixels[j, i] = (0, 0, 0)
            elif maze[i, j] == 1:
                pixels[j, i] = (0, 255, 0)
    for x, y in path:
        pixels[y, x] = (255, 0, 0)
    return np.array(img)

if __name__ == "__main__":
    maze_size = 8
    input_shape = (maze_size,maze_size,1)
    num_actions = 4
    agent = DQNAgent(input_shape,num_actions)
    env = MazeEnv(maze_size)
    for epoch in range(100):
        steps = agent.train(env,50)

        plt.imshow(maze_to_image(env.maze,[]))
        plt.savefig(f"figs/maze_{epoch+1}.png") # 保存迷宫原始图像
        plt.clf()
 
        plt.plot(steps)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Training Steps')
        plt.savefig(f"figs/train_{epoch+1}.png") # 保存训练图像
        plt.clf()
 
        solve = maze_to_image(env.maze,env.path)
 
        plt.imshow(solve)
        plt.savefig(f"figs/sloves_{epoch+1}.png") # 保存最后一次路线
        plt.clf()
 
    env.reset(complete=True)  # 完全重置环境
    agent.save("model")