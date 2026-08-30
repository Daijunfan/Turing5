from dataclasses import dataclass
from fractions import Fraction

from tcmk.dynamics import offline_optimum
from tcmk.counterexamples import find_local_context_counterexample
from tcmk.kernel import ce_tcmk, minimum_kernel_oracle
from tcmk.residual import exact_separation


@dataclass(frozen=True)
class TinyTask:
    kind: int


class TinyModel:
    # State 2 is a transition bridge between service specialists 0 and 1.
    service = ((0, 8, 3), (8, 0, 3))
    move = (
        (0, 10, 1),
        (10, 0, 1),
        (1, 1, 0),
    )

    @property
    def state_count(self):
        return 3

    def service_cost(self, task, state):
        return Fraction(self.service[task.kind][state])

    def transition_cost(self, task, previous, state):
        del task
        return Fraction(2 if previous is None else self.move[previous][state])


def test_residual_search_matches_direct_word_enumeration():
    model = TinyModel()
    alphabet = (TinyTask(0), TinyTask(1))
    result = exact_separation(model, alphabet, 4, (0, 1))
    assert result.gap is not None and result.gap > 0
    full = offline_optimum(model, result.trace)
    kernel = offline_optimum(model, result.trace, (0, 1))
    assert kernel.cost - full.cost == result.gap


def test_ce_tcmk_and_minimum_oracle_are_exact():
    model = TinyModel()
    alphabet = (TinyTask(0), TinyTask(1))
    ce = ce_tcmk(model, alphabet, 4, (0, 1))
    minimum = minimum_kernel_oracle(model, alphabet, 4)
    assert ce.final_separation.equivalent
    assert minimum.final_separation.equivalent
    assert len(ce.kernel) >= len(minimum.kernel)


def test_one_occurrence_context_rule_has_a_consecutive_stay_counterexample():
    counterexample = find_local_context_counterexample()
    assert tuple(task.index for task in counterexample.trace) == (1, 1, 0, 0)
    assert counterexample.pruned_cost - counterexample.full_cost == 1
