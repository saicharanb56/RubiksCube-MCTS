import logging
import os
import threading
from collections import defaultdict
from copy import copy, deepcopy
from typing import List, Tuple, Union

import argparse
import numpy as np
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from numpy.typing import NDArray
from rubikscube import Cube
from torch.distributed.rpc import RRef, remote, rpc_async, rpc_sync

from model import NNet, ResnetModel

parser = argparse.ArgumentParser()

parser.add_argument('--load_path', type=str)
parser.add_argument('--c', default=5.0, type=float)
parser.add_argument('--mu', default=1.0, type=float)
parser.add_argument('--network', default='simple', type=str)

args = parser.parse_args()

NUM_ACTIONS = 12
INT_TO_ACTIONS = Cube.cube_qtm().get_turns()
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"

logging.basicConfig(level=logging.INFO)


def softmax(x):
    x_max = np.max(x)
    return (np.exp(x - x_max) / np.exp(x - x_max).sum())


class State:
    def __init__(self,
                 logit: np.ndarray,
                 value: float,
                 cube_state,
                 representation: Tuple[int, ...],
                 c=args.c) -> None:
        self._representation: Tuple[int, ...] = copy(representation)
        self._cube_state = copy(cube_state)

        self.N: np.ndarray = np.zeros(NUM_ACTIONS, dtype=np.int32)
        self.N_sum: int = 0
        self.W: np.ndarray = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.L: np.ndarray = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.P: np.ndarray = softmax(logit.copy())
        self.value = value
        self.c = c

        self.dirichlet_epsilon = 0.25
        self.dirichlet_alpha = 0.01

    def P_pertubed(self):
        noise = np.random.dirichlet([self.dirichlet_alpha] * NUM_ACTIONS)
        return (1 - self.dirichlet_epsilon
                ) * self.P + self.dirichlet_epsilon * noise

    def U(self) -> np.ndarray:
        return self.c * self.P_pertubed() * np.sqrt(self.N_sum) / (1 + self.N)

    def Q(self) -> np.ndarray:
        return self.W - self.L

    def __hash__(self) -> int:
        return hash(self._representation)


class Node:
    def __init__(self, p, v, cube_state, repr, terminal) -> None:
        self._state = State(p, v, cube_state, repr)
        self.terminal = terminal
        self.mu = args.mu

        if not self.terminal:
            self._children = [None] * NUM_ACTIONS
            self.connected_to_terminal = np.zeros(NUM_ACTIONS, dtype=bool)
            self.leaf_node = True

    def children(self):
        return ((i, child) for (i, child) in enumerate(self._children)
                if child is not None)

    def U(self) -> np.ndarray:
        return self._state.U()

    def Q(self) -> np.ndarray:
        return self._state.Q()

    def get_value(self) -> float:
        return self._state.value

    def __hash__(self):
        return hash(self._state)

    def get_cube_state(self):
        return self._state._cube_state

    def add_child(self, child_node, action):
        self._children[action] = child_node

    def update_count_action(self, action):
        self._state.N[action] += 1
        self._state.N_sum += 1

    def update_action_value(self, action, value):
        self._state.W[action] = max(self._state.W[action], value)

    def increase_virtual_action_value(self, action):
        self._state.L[action] += self.mu

    def decrease_virtual_action_value(self, action):
        self._state.L[action] -= self.mu


class MCTSAgent:
    def __init__(self, model: NNet) -> None:
        self.nodes = dict()
        self.model = model
        self._shortest_path = None

    def solve(self,
              cube_state,
              number_iterations: int = 100,
              max_depth=50) -> bool:
        """
        performs MCTS tree search to find path to solved cube
        """
        # set maximum possible solution length
        self._shortest_path = max_depth

        # set cube state
        cube = Cube.cube_qtm()
        cube.set_state(cube_state)

        # initialize root node
        p, v, cube_state, representation, terminal = self.cube_data(cube)
        root = Node(p, v, cube_state, repr, terminal)
        root.leaf_node = False

        # perform number_iterations of search with max_depth
        for i in range(number_iterations):
            logging.debug(f"Searching iteration {i}")
            _, flag = self.mcts_search(root, max_depth)
            if flag:
                return bfs(root)

        return None

    def get_child(self, node, action):

        if node._children[action]:  # if child already exists
            return node._children[action]

        # initialize cube and perfom best action
        cube = Cube.cube_qtm()
        cube.set_state(node.get_cube_state())
        cube.turn(action)

        # check if node already exists
        representation = tuple(cube.representation())

        if representation in self.nodes:  # node already exists
            child_node = self.nodes[representation]
            node.add_child(child_node, action)  # add as child to current node
            return child_node

        # create new node
        p, v, cube_state, _, terminal = self.cube_data(cube)
        new_node = Node(p, v, cube_state, representation, terminal)

        node.add_child(new_node, action)  # add as child to current node
        self.nodes[representation] = new_node  # add to dict nodes

        return new_node

    def cube_data(self, cube):
        # get cube's state information
        representation = cube.representation()
        cube_state = cube.get_state()
        terminal = cube.solved()

        # compute value and policy for cube state
        with torch.no_grad():
            p, v = self.model(torch.tensor(representation,
                                           dtype=torch.float32))

        return p.cpu().numpy(), v.cpu().numpy(), cube_state, tuple(
            representation), terminal

    def mcts_search(self, node: Node, max_depth) -> Tuple[float, bool]:

        # check if terminal state is reached
        if node.terminal:
            return 0, True

        if node.leaf_node:
            node.leaf_node = False
            return node.get_value(), False

        # check if the maximum depth has been reached or node is a leaf node
        if max_depth == 0:
            return node.get_value(), False

        action = self.best_action(node)
        node.update_count_action(action)
        node.increase_virtual_action_value(action)

        child_action_value, reached_terminal = self.mcts_search(
            self.get_child(node, action), max_depth - 1)

        if reached_terminal:
            node.connected_to_terminal[action] = True

        node.update_action_value(action, child_action_value)
        node.decrease_virtual_action_value(action)

        return node.get_value(), reached_terminal

    def best_action(self, node: Node) -> None:

        if node._state.N_sum == 0:
            return np.argmax(node._state.P_pertubed())

        return np.argmax(node.U() + node.Q())


def bfs(root: Node) -> Union[None, List[int]]:
    '''
    Performs bfs from a given node to find the shortest solution in the mcts search tree. 
    '''
    # keep track of explored states for bfs
    visited = defaultdict(lambda: False)

    # queue for bfs
    queue = [[(root, None)]]

    while queue:

        # pop the first in the queue
        solve = queue.pop(0)
        (node, _) = solve[-1]

        if not visited[node]:
            # visit the children visited during mcts
            for action, child in node.children():
                # update current solution
                new_solve = list(solve) + [(child, action)]
                queue.append(new_solve)

                # check is terminal state is reached and return the
                # solution
                if child.terminal:
                    return [
                        move for (_, move) in new_solve if move is not None
                    ]

            visited[node] = True

    return None


NODES = dict()


class MCTSAsyncAgent(MCTSAgent):
    def __init__(self) -> None:
        super().__init__(None)

        self.id = rpc.get_worker_info().id - 1

    def cube_data_async(self, agent_rref, cube):
        # get cube's state information
        representation = cube.representation()
        cube_state = cube.get_state()
        terminal = cube.solved()

        # compute value and policy for cube state
        p, v = rpc.rpc_sync(agent_rref.owner(),
                            MCTSAsyncHandler.select_action,
                            args=(agent_rref, self.id, representation))

        return p, v, cube_state, tuple(representation), terminal

    def get_child_async(self, agent_rref, node, action):

        if node._children[action]:  # if child already exists
            return node._children[action]

        # initialize cube and perfom best action
        cube = Cube.cube_qtm()
        cube.set_state(node.get_cube_state())
        cube.turn(action)

        # check if node already exists
        representation = tuple(cube.representation())

        if representation in NODES:  # node already exists
            child_node = NODES[representation]
            node.add_child(child_node, action)  # add as child to current node
            return child_node

        # create new node
        p, v, cube_state, _, terminal = self.cube_data_async(agent_rref, cube)
        new_node = Node(p, v, cube_state, representation, terminal)

        node.add_child(new_node, action)  # add as child to current node
        NODES[representation] = new_node  # add to dict nodes

        return new_node

    def mcts_search_async(self, agent_rref, node: Node,
                          max_depth) -> Tuple[float, bool]:

        # check if terminal state is reached
        if node.terminal:
            return 0, True

        if node.leaf_node:
            node.leaf_node = False
            return node.get_value(), False

        # check if the maximum depth has been reached or node is a leaf node
        if max_depth == 0:
            return -np.inf, False

        action = self.best_action(node)
        node.update_count_action(action)
        node.increase_virtual_action_value(action)

        child_action_value, reached_terminal = self.mcts_search_async(
            agent_rref, self.get_child_async(agent_rref, node, action),
            max_depth - 1)

        if reached_terminal:
            node.connected_to_terminal[action] = True

        node.update_action_value(action, child_action_value)
        node.decrease_virtual_action_value(action)

        return node.get_value(), reached_terminal


class MCTSAsyncHandler:
    def __init__(self, model, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.model = model

        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, MCTSAsyncAgent, args=()))

        self.states = torch.zeros(len(self.ob_rrefs), 480)

        self.future_actions = torch.futures.Future()
        self.lock = threading.Lock()
        self.pending_states = len(self.ob_rrefs)

    @staticmethod
    @rpc.functions.async_execution
    def select_action(agent_rref, ob_id, state):

        self = agent_rref.local_value()
        self.states[ob_id].copy_(torch.tensor(state, dtype=torch.float32))

        def future_callback(future):
            probs, values = future.wait()
            return probs[ob_id], values[ob_id]

        future_action = self.future_actions.then(future_callback)

        with self.lock:
            self.pending_states -= 1
            if self.pending_states == 0:
                self.pending_states = len(self.ob_rrefs)
                with torch.no_grad():
                    probs, value = self.model(self.states)
                future_actions = self.future_actions
                self.future_actions = torch.futures.Future()
                future_actions.set_result(
                    (probs.cpu().numpy(), value.cpu().numpy()))

        return future_action

    def cube_data(self, cube):
        # get cube's state information
        representation = cube.representation()
        cube_state = cube.get_state()
        terminal = cube.solved()

        # compute value and policy for cube state
        with torch.no_grad():
            p, v = self.model(torch.tensor(representation,
                                           dtype=torch.float32))

        return p.cpu().numpy(), v.cpu().numpy(), cube_state, tuple(
            representation), terminal

    def solve(self,
              cube_state,
              number_iterations: int = 1000,
              max_depth=100) -> bool:
        """
        performs MCTS tree search to find path to solved cube
        """
        # set cube state
        cube = Cube.cube_qtm()
        cube.set_state(cube_state)

        # initialize root node
        p, v, cube_state, representation, terminal = self.cube_data(cube)

        if terminal:
            return []

        root = Node(p, v, cube_state, repr, terminal)
        NODES[representation] = root
        root.leaf_node = False

        # perform number_iterations of search with max_depth
        for i in range(number_iterations):
            futs = []
            for ob_rref in self.ob_rrefs:
                futs.append(ob_rref.rpc_async().mcts_search_async(
                    self.agent_rref, root, max_depth))

            rets = torch.futures.wait_all(futs)

            if np.any([flag for (_, flag) in rets]):
                return bfs(root)

        return None


def solve(rank,
          world_size,
          state,
          model,
          number_iterations=1000,
          max_depth=100):

    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '29507'

    if rank == 0:  # agent

        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

        agent = MCTSAsyncHandler(model, world_size)

        solution = agent.solve(state, number_iterations, max_depth)

        if solution is not None:
            logging.info(
                f"solution ::: {[INT_TO_ACTIONS[action] for action in solution]}"
            )
        else:
            logging.info("Failed to solve the cube")

    else:

        rpc.init_rpc(OBSERVER_NAME.format(rank),
                     rank=rank,
                     world_size=world_size)

    rpc.shutdown()


if __name__ == '__main__':

    if args.network == 'simple':
        model = NNet()
    else:
        model = ResnetModel(batch_norm=False)
    
    checkpoint = torch.load(args.load_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # np.random.seed(17)

    cube = Cube.cube_qtm()
    scramble_list = np.random.choice(cube.get_turns(), 17)
    logging.info(f"scramble list ::: {scramble_list}")

    for m in scramble_list:
        cube.turn(int(m))

    state = cube.get_state()

    # world_size = 12
    #
    # mp.spawn(solve,
    #          args=(world_size, state, model),
    #          nprocs=world_size,
    #          join=True)

    mcts = MCTSAgent(model)

    solution = mcts.solve(state, number_iterations=100000, max_depth=50)

    if solution is not None:
        logging.info(
            f"solution ::: {[INT_TO_ACTIONS[action] for action in solution]}")
    else:
        logging.info("Failed to solve the cube")
