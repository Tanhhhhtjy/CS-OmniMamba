import os

from omnimamba import viz


class DummyDataset:
    def __init__(self, pwv_paths):
        self.pwv_paths = list(pwv_paths)

    def __len__(self):
        return len(self.pwv_paths)


def test_resolve_viz_indices_preserves_requested_order():
    paths = [
        os.path.join("data", "PWV", "2023-08-04-06-00-00.png"),
        os.path.join("data", "PWV", "2023-08-05-00-00-00.png"),
        os.path.join("data", "PWV", "2023-08-06-11-00-00.png"),
    ]
    dataset = DummyDataset(paths)
    requested = [
        "2023-08-05-00-00-00",
        "2023-08-04-06-00-00.png",
        "2023-08-06-11-00-00",
    ]

    indices = viz._resolve_viz_indices(dataset, requested)

    assert indices == [1, 0, 2]


def test_resolve_viz_indices_missing_times_are_ignored():
    paths = [
        os.path.join("data", "PWV", "2023-08-04-06-00-00.png"),
        os.path.join("data", "PWV", "2023-08-05-00-00-00.png"),
    ]
    dataset = DummyDataset(paths)

    indices = viz._resolve_viz_indices(dataset, ["2023-08-06-11-00-00"])

    assert indices == []
