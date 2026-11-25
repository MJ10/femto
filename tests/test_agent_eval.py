"""The terminal tasks' checks (and partial-credit parts) pass with their reference solutions and fail when nothing is
done."""

from types import SimpleNamespace

import terminal_tasks
from tools import Sandbox


def test_terminal_task_checks():
    generators = terminal_tasks.GENERATORS + terminal_tasks.HELDOUT_GENERATORS
    for task in terminal_tasks.tasks(seed=1, per_generator=3, generators=generators):
        with Sandbox(task.files) as box:
            nothing = SimpleNamespace(answer="I don't know.", last_output=None, sandbox=box)
            assert not task.check(nothing) and (task.parts is None or task.parts(nothing) == 0), task.name
            output, code = box.run("bash", task.solution)
            assert code == 0, (task.name, output)
            solved = SimpleNamespace(answer="", last_output=output, sandbox=box)
            assert task.check(solved) and (task.parts is None or task.parts(solved) == 1), (task.name, output)
