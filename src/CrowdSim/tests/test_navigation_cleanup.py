"""A failed navigation constructor still releases any log it opened."""

from CrowdSim.nav_manager import CrowdNavigationManager


def test_cleanup_before_logs_are_initialized():
    nav = CrowdNavigationManager.__new__(CrowdNavigationManager)
    nav._close_trajectory_logs()


def test_cleanup_with_only_first_log_initialized(tmp_path):
    nav = CrowdNavigationManager.__new__(CrowdNavigationManager)
    path = tmp_path / "trajectory.jsonl"
    nav._trajectory_log_file = path.open("w")
    nav._trajectory_log_file.write('{"agent": 0}\n')
    nav._close_trajectory_logs()
    nav._close_trajectory_logs()
    assert nav._trajectory_log_file.closed
    assert path.read_text() == '{"agent": 0}\n'
