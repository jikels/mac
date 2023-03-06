from mbmrl_torch.gym.envs.mujoco.halfcheetah_base import HalfCheetahBase
from mbmrl_torch.gym.envs.meta_env import MetaEnv
import numpy as np

class HalfCheetahHFieldEnv(HalfCheetahBase,MetaEnv):
    
    def __init__(self, task={'hfield': None}):
        # init env
        xml_path = 'mbmrl_torch/gym/envs/mujoco/assets/half_cheetah_hfield.xml'
        super(HalfCheetahHFieldEnv, self).__init__(xml_path)

        # init task
        self.init_hfield_size = self.model.hfield_size[0].copy()
        self.init_hfield_data = self.model.hfield_data[:].copy()

        self.x_walls = np.array([250, 260, 261, 270, 280, 285])
        self.height_walls = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.height = 0.8
        self.width = 15

        self.task = task

    # -------- MetaEnv Methods --------
    def set_task(self, task):
        task = task['hfield']
        assert task in [None, 'hfield', 'hill', 'basin', 'steep', 'gentle']
        self.__init__(task)

        self.model.hfield_size[0] = self.init_hfield_size
        self.model.hfield_data[:] = self.init_hfield_data
        
        if self.task == 'hfield':
            height = np.random.uniform(0.2, 1)
            width = 10
            n_walls = 6
            self.model.hfield_size[0] = np.array([50, 5, height, 0.1])
            x_walls = np.random.choice(np.arange(255, 310, width), replace=False, size=n_walls)
            x_walls.sort()
            sign = np.random.choice([1, -1], size=n_walls)
            sign[:2] = 1
            height_walls = np.random.uniform(0.2, 0.6, n_walls) * sign
            row = np.zeros((500,))
            for i, x in enumerate(x_walls):
                terrain = np.cumsum([height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x+width:] = row[x+width - 1]
            row = (row - np.min(row))/(np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            hfield = hfield.flatten()
            self.model.hfield_data[:] = hfield

        elif self.task == 'basin':
            self.height_walls = np.array([-1, 1, 0., 0., 0., 0.])
            self.height = 0.55
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size[0] = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            hfield = hfield.flatten()
            self.model.hfield_data[:] = hfield

        elif self.task == 'hill':
            self.height_walls = np.array([1, -1, 0, 0., 0, 0])
            self.height = 0.6
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size[0] = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            hfield = hfield.flatten()
            self.model.hfield_data[:] = hfield

        elif self.task == 'gentle':
            self.height_walls = np.array([1, 1, 1, 1, 1, 1])
            self.height = 1
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size[0] = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            hfield = hfield.flatten()
            self.model.hfield_data[:] = hfield

        elif self.task == 'steep':
            self.height_walls = np.array([1, 1, 1, 1, 1, 1])
            self.height = 4
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size[0] = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            hfield = hfield.flatten()
            self.model.hfield_data[:] = hfield

        elif self.task == None:
            pass
 
    def sample_tasks(self, num_tasks):
        t = ['hfield', 'hill', 'basin', 'steep', 'gentle']
        hfields = np.random.choice(t,num_tasks)
        tasks = [{'hfield': hfield} for hfield in hfields]
        return tasks

