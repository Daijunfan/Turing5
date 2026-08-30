import numpy as np

from morphn.dag import DagTask, SharedDag, checkpoint_architectures


def test_actual_shared_dag_execution_matches_reference():
    dag = SharedDag(128, width=16)
    architectures = checkpoint_architectures(128)
    cache = {}
    for index, target in enumerate((127, 96, 126, 80, 127)):
        task = DagTask(target, 128)
        run = dag.execute(task, cache, architectures[index % len(architectures)], actual=True)
        cache = run.cache
        assert np.array_equal(run.output, dag.reference(target))
