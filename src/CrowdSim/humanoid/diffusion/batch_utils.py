"""Current unicycle state in the agent-centred TRACE frame."""

class _BatchUtils:
    @staticmethod
    def get_current_states(batch, dyn_type=None):
        speed = batch["curr_speed"]
        result = speed.new_zeros((speed.shape[0], 4))
        result[:, 2] = speed
        return result


def batch_utils():
    return _BatchUtils()
