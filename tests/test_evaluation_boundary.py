"""Exercise only the public demo-policy environment and client interfaces."""
from PhysicalRSI_baselines.robodojo.evaluation import eval_one_episode_batch


def test_batch_removes_finished_environments_without_terminal_frame_probe():
    class Environment:
        def __init__(self):
            self.steps = {2: 0, 7: 0}
            self.actions = []
        def get_running_env_idx_list(self):
            return [i for i, step in self.steps.items() if step < (1 if i == 2 else 3)]
        def is_episode_end(self):
            return not self.get_running_env_idx_list()
        def get_obs_batch(self, indices):
            assert indices and set(indices) <= set(self.get_running_env_idx_list())
            return [{'env_idx': i, 'step': self.steps[i]} for i in indices]
        def take_action_batch(self, actions, indices):
            assert indices == self.get_running_env_idx_list()
            self.actions.append((indices, actions))
            for i in indices:
                self.steps[i] += 1
    class Client:
        def __init__(self):
            self.observations = []
            self.resets = 0
        def call(self, func_name, obs=None):
            if func_name == 'reset':
                self.resets += 1
            elif func_name == 'update_obs_batch':
                self.observations.extend((r['env_idx'], r['step']) for r in obs)
            elif func_name == 'get_action_batch':
                return [[{'value': i}] * 2 for i in obs]
            else:
                raise AssertionError(func_name)
    env, client = Environment(), Client()
    eval_one_episode_batch(env, client)
    assert client.resets == 1
    assert client.observations == [(2, 0), (7, 0), (7, 1), (7, 2)]
    assert [indices for indices, _ in env.actions] == [[2, 7], [7], [7]]


def test_frozen_collector_compatibility_never_accesses_private_transport():
    import pytest
    from PhysicalRSI_baselines.robodojo.evaluation import configure_transport
    class Client:
        @property
        def _client(self):
            raise AssertionError('Private transport accessed')
    client = Client()
    with pytest.warns(DeprecationWarning, match='no longer modifies'):
        assert configure_transport(client, 1800) is client
    assert vars(client) == {}
