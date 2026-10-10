"""Bounded real-source branch execution with stdlib spies; no tensors/NN/physics."""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / 'output/isolated_training/mixed_state_terrain_state_prior_stratified_matched_v1/code'


def record(path):
    raw = Path(path).read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class Value:
    def __init__(self, data, observation_stamp):
        self.data = deepcopy(data)
        self.observation_stamp = observation_stamp

    def clone(self):
        return deepcopy(self)

    def detach(self):
        return self


class World:
    def __init__(self, terminal=False):
        self.time = 0
        self.amp_history = Value(tuple(range(0, -10, -1)), 0)
        self.amp_history_episode_ids = Value((7,) * 10, 0)
        self.episode_ids = Value((7,), 0)
        self.terminal = terminal
        self.events = []

    def step(self, applied):
        self.events.append('physics_step')
        self.time += 1
        self.amp_history = Value((self.time,) + self.amp_history.data[:-1], self.time)
        self.amp_history_episode_ids = Value((7,) * 10, self.time)
        return None, Value(self.terminal, self.time), Value(False, self.time)

    def reset(self):
        self.events.append('reset')
        self.amp_history = Value((999,) * 10, 999)
        self.amp_history_episode_ids = Value((8,) * 10, 999)
        self.episode_ids = Value((8,), 999)


class Replay:
    def add(self, history, actions, *, keep_probability):
        self.history = history.clone()
        self.actions = actions.clone()
        self.keep_probability = keep_probability


class Trainer:
    def __init__(self, world, mode):
        self.world = world
        self.config = SimpleNamespace(discriminator_input=mode, discriminator_replay_keep_probability=.01)
        self.calls = []
        self.policy_replay = Replay()

    def act(self, *args, **kwargs):
        return {key: Value(.125, 0) for key in ('actions', 'applied_actions', 'log_prob', 'values')}

    def amp_reward(self, history, actions):
        self.world.events.append('amp_reward')
        self.calls.append({'history': history.clone(), 'action': actions.clone()})
        return Value(history.data[0], history.observation_stamp)

    def value(self, *args):
        return Value(0, 1)


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def extract(path):
    text = Path(path).read_text()
    tree = ast.parse(text)
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
             and n.target.id == 'step' and ast.unparse(n.iter) == 'range(length)']
    assert len(loops) == 1
    loop = loops[0]
    nodes = []
    for node in loop.body:
        if any(assigned(node, key) for key in ('history', 'history_episode', 'episode', 'action', 'amp_reward')):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'world.step':
            nodes.append(node)
        elif isinstance(node, ast.If) and ast.unparse(node.test) == "not pure_evaluation and trainer.config.discriminator_input == 'state'":
            nodes.append(node)
    transitions = [n for n in ast.walk(loop) if assigned(n, 'transition')]
    assert len(transitions) == 1
    reward_blocks = [n for n in nodes if isinstance(n, ast.If)]
    reset_calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'world.reset']
    assert reset_calls and all(transitions[0].lineno < n.lineno for n in reset_calls)
    if reward_blocks:
        assert nodes.index(reward_blocks[0]) > next(i for i, n in enumerate(nodes)
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'world.step')
        assert reward_blocks[0].end_lineno < transitions[0].lineno
    return nodes, transitions[0]


def run_case(path, *, mode='state', terminal=False, evaluation=False, omit_post_history=False):
    nodes, transition = extract(path)
    nodes = deepcopy(nodes)
    if omit_post_history:
        for n in nodes:
            if isinstance(n, ast.If):
                n.body = [c for c in n.body if not assigned(c, 'history')]
    world = World(terminal)
    trainer = Trainer(world, mode)
    env = {'world': world, 'trainer': trainer, 'pure_evaluation': evaluation,
           'args': SimpleNamespace(evaluate_steps=int(evaluation)),
           'self_obs': Value(0, 0), 'task_obs': Value(0, 0)}
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, type_ignores=[])), '<actual runner selected statements>', 'exec'), env)
    expected_time = 1 if mode == 'state' and path == HERE / 'train_paper_amp.py' else 0
    if evaluation:
        assert env['amp_reward'] is None and trainer.calls == []
        return {'case': 'evaluation', 'reward_calls': 0, 'runner_transition_not_consumed': True}
    assert len(trainer.calls) == 1
    assert trainer.calls[0]['history'].data[0] == expected_time
    assert env['history'].data[0] == expected_time
    assert env['history_episode'].observation_stamp == expected_time
    assert env['amp_reward'].data == expected_time
    env.update({'next_self': Value(0, 1), 'next_task': Value(0, 1), 'reward': Value(0, 1),
                'components': {key: Value(0, 1) for key in ('trajectory', 'energy', 'direction', 'signed_power_w')},
                'forward_velocity': Value(0, 1)})
    exec(compile(ast.fix_missing_locations(ast.Module([transition], type_ignores=[])), '<actual transition dictionary>', 'exec'), env)
    original = env['transition']
    collected = {key: value.detach().clone() for key, value in original.items()}
    source = ast.parse((PARENT / 'CrowdSim/humanoid/training/trainer.py').read_text())
    replay = [n for n in ast.walk(source) if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.policy_replay is not None'
              and any(isinstance(c, ast.Expr) and isinstance(c.value, ast.Call)
                      and ast.unparse(c.value.func) == 'self.policy_replay.add' for c in n.body)]
    assert len(replay) == 1
    exec(compile(ast.fix_missing_locations(ast.Module(replay, type_ignores=[])), '<actual trainer replay handoff>', 'exec'),
         {'self': trainer, 'flat': collected, 'config': trainer.config})
    assert trainer.policy_replay.history.data == trainer.calls[0]['history'].data
    assert trainer.policy_replay.history.observation_stamp == env['history_episode'].observation_stamp == expected_time
    world.reset()
    assert collected['amp_history'].data[0] == expected_time
    assert collected['amp_rewards'].data == expected_time
    assert collected['episode_ids'].data == (7,) and collected['terminated'].data is terminal
    assert trainer.policy_replay.history.data[0] == expected_time and world.amp_history.data[0] == 999
    return {'case': mode + ('_terminal' if terminal else '_nonterminal'), 'reward_history_stamp': expected_time,
            'rollout_and_replay_exact_same_history': True, 'terminal_preserved_before_reset': terminal,
            'reset_does_not_mutate_cloned_records': True, 'events': world.events}


def main():
    cases = [run_case(HERE / 'train_paper_amp.py'), run_case(HERE / 'train_paper_amp.py', terminal=True),
             run_case(PARENT / 'scripts/train_paper_amp.py'), run_case(HERE / 'train_paper_amp.py', mode='state_action'),
             run_case(HERE / 'train_paper_amp.py', evaluation=True)]
    try:
        run_case(HERE / 'train_paper_amp.py', omit_post_history=True)
    except AssertionError:
        negative = 'missing actual post-step history reassignment detected'
    else:
        raise AssertionError('Necessary mismatch negative did not fail')
    report = {'schema': 'state_only_q.actual_poststep_timing_spy_CPU.v1', 'status': 'passed',
              'scope': 'Executes selected actual runner AST statements, full actual transition dictionary and actual trainer replay handoff with stdlib spy objects. Not tensor, PPO, discriminator, replay-ringbuffer, physics or model execution.',
              'cases': cases, 'necessary_negative': negative,
              'source': {'runner': record(HERE / 'train_paper_amp.py'), 'parent_runner': record(PARENT / 'scripts/train_paper_amp.py'),
                         'trainer': record(PARENT / 'CrowdSim/humanoid/training/trainer.py'), 'checker': record(__file__)},
              'Torch_NN_SDK_GPU_or_dataset_used': False, 'physical_or_style_gain_proven': False}
    with (HERE / 'actual_timing_CPU.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': 'passed', 'positive_cases': len(cases), 'negative_detected': True}))


if __name__ == '__main__':
    main()
