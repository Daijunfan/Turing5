from random import Random

from tcmk.certificate_checker import check_migration_certificate
from tcmk.matrix_runtime import PersistentExecutor, random_matrix
from tcmk.morphology import canonical_morphologies
from tcmk.dynamics import offline_optimum
from tcmk.kernel import ce_tcmk
from tcmk.morphology import static_pareto
from tcmk.real_migration import (
    PersistentMatrixModel,
    RealTask,
    exact_migration_totals,
    plan_migration,
)


def test_certificates_and_continuous_real_updates():
    dims = (2, 3, 2, 4, 2, 3)
    states = canonical_morphologies(dims)
    random = Random(20260830)
    leaves = [
        random_matrix(dims[index], dims[index + 1], random)
        for index in range(len(dims) - 1)
    ]
    executor = PersistentExecutor(dims, leaves, states[0].tree)
    for step in range(60):
        update = random.randrange(len(leaves))
        target = states[random.randrange(len(states))].tree
        certificate = plan_migration(dims, executor.tree, target, update)
        totals = exact_migration_totals(dims, executor.tree, target, update)
        assert (
            certificate.peak_memory,
            certificate.total_work,
            certificate.read_bytes,
            certificate.write_bytes,
        ) == (
            totals.peak_memory,
            totals.total_work,
            totals.read_bytes,
            totals.write_bytes,
        )
        assert certificate.exact
        assert check_migration_certificate(
            dims, executor.tree, target, certificate
        ).valid
        replacement = random_matrix(dims[update], dims[update + 1], random)
        executor.apply_update(update, replacement, target, certificate)
    assert executor.version == 60


def test_static_pareto_is_dynamically_incomplete_in_real_reuse_model():
    dims = (1, 2, 4, 1, 6, 6)
    states = canonical_morphologies(dims)
    static_ids = tuple(states.index(state) for state in static_pareto(states))
    model = PersistentMatrixModel(dims, states)
    trace = tuple(
        RealTask(index, 10**12) for index in (0, 4, 4, 2, 4, 2)
    )
    full = offline_optimum(model, trace)
    pruned = offline_optimum(model, trace, static_ids)
    assert full.cost == 594
    assert pruned.cost == 738
    assert any(state not in static_ids for state in full.path)


def test_real_model_ce_tcmk_certificate_through_horizon_four():
    dims = (1, 2, 4, 1, 6, 6)
    states = canonical_morphologies(dims)
    static_ids = tuple(states.index(state) for state in static_pareto(states))
    model = PersistentMatrixModel(dims, states)
    alphabet = tuple(RealTask(index, 10**12) for index in range(5))
    certificate = ce_tcmk(model, alphabet, 4, static_ids)
    assert certificate.final_separation.equivalent
    assert len(certificate.kernel) == 4
    assert len(certificate.kernel) < len(states)
