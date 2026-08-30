universe u v

namespace MorphN

structure StepResult (Residual : Type u) where
  increment : Int
  next : Residual
deriving Repr, DecidableEq

def run {Residual : Type u} {Task : Type v}
    (step : Residual → Task → StepResult Residual)
    (initial : Residual) : List Task → Int × Residual
  | [] => (0, initial)
  | task :: suffix =>
      let first := step initial task
      let later := run step first.next suffix
      (first.increment + later.1, later.2)

def increments {Residual : Type u} {Task : Type v}
    (step : Residual → Task → StepResult Residual)
    (initial : Residual) : List Task → List Int
  | [] => []
  | task :: suffix =>
      let first := step initial task
      first.increment :: increments step first.next suffix

theorem same_residual_same_run
    {Residual : Type u} {Task : Type v}
    (step : Residual → Task → StepResult Residual)
    {left right : Residual}
    (equalResidual : left = right)
    (suffix : List Task) :
    run step left suffix = run step right suffix := by
  subst right
  rfl

theorem same_residual_same_increment_stream
    {Residual : Type u} {Task : Type v}
    (step : Residual → Task → StepResult Residual)
    {left right : Residual}
    (equalResidual : left = right)
    (suffix : List Task) :
    increments step left suffix = increments step right suffix := by
  subst right
  rfl

theorem equal_history_residuals_determine_all_future_costs
    {Residual : Type u} {Task : Type v}
    (step : Residual → Task → StepResult Residual)
    (residualOf : List Task → Residual)
    {history₁ history₂ : List Task}
    (equalResidual : residualOf history₁ = residualOf history₂)
    (future : List Task) :
    run step (residualOf history₁) future =
      run step (residualOf history₂) future ∧
    increments step (residualOf history₁) future =
      increments step (residualOf history₂) future := by
  constructor
  · exact same_residual_same_run step equalResidual future
  · exact same_residual_same_increment_stream step equalResidual future

end MorphN
