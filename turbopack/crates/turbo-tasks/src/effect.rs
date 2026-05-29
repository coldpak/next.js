use std::{
    collections::hash_map,
    error::Error as StdError,
    future::Future,
    mem::{forget, replace},
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::Result;
use futures::{StreamExt, TryStreamExt};
use parking_lot::{Mutex, MutexGuard};
use rustc_hash::FxHashMap;
use tracing::Instrument;

use crate::{
    self as turbo_tasks, CollectiblesSource, NonLocalValue, ResolvedVc, TryJoinIterExt, emit,
    event::Event,
    invalidation::{Invalidator, get_invalidator},
    manager::{
        debug_assert_in_top_level_task, debug_assert_not_in_top_level_task, with_turbo_tasks,
    },
    trace::TraceRawVcs,
};

const APPLY_EFFECTS_CONCURRENCY_LIMIT: usize = 1024;

/// Emit-time effect. Lives inside an [`EffectInstance`] cell and is allowed to read Vcs in
/// [`Effect::capture`] (since `capture` runs from within a turbo-tasks task during
/// [`take_effects`]).
///
/// Implementations resolve any `ResolvedVc`/`Vc` data they need inside `capture()` and return a
/// [`CapturedEffect`] holding only the pre-resolved fields. The captured value is what gets
/// stored in [`Effects`] and what eventually drives the side effect at apply time. After a
/// successful [`Effects::apply`], the captured Vec is dropped — releasing any `ReadRef`s the
/// `CapturedEffect`s held and breaking the strong-count cascade onto upstream cells.
pub trait Effect: TraceRawVcs + NonLocalValue + Send + Sync + 'static {
    /// The pre-resolved companion that performs the side effect. See [`CapturedEffect`].
    type Captured: CapturedEffect;

    /// Resolve any Vc/ReadRef data needed for `apply()`. Runs inside the turbo-tasks task
    /// context of [`take_effects`], so it may `.await` Vc reads. The returned
    /// [`CapturedEffect`] must hold only pre-resolved data — its `apply()` will run from a
    /// top-level task where Vc reads are not expected.
    fn capture(&self) -> impl Future<Output = Result<Self::Captured>> + Send;
}

/// Post-capture effect. Holds only pre-resolved data and performs the actual side effect.
///
/// `apply()` is responsible for coordinating with [`EffectStateStorage`] via
/// [`EffectStateStorage::run_apply`] (which handles the per-key state machine, in-progress
/// coordination, dedup-hit short-circuit, and panic recovery). Implementations must not read Vcs
/// at apply time — the captured form holds only pre-resolved data.
///
/// An implementation may choose to elide content materialization in [`Effect::capture`] when the
/// storage state already holds a matching `Applied { value_hash }`. In that case, the captured
/// form has no content to apply, and `apply()` must return [`ApplyOutcome::Retry`] when the state
/// machine forces an actual apply (storage was stomped between capture and apply). `Effects` will
/// collect the Retry signal, invalidate the producing task, and return [`EffectsError::Retry`].
pub trait CapturedEffect: TraceRawVcs + NonLocalValue + Send + Sync + 'static {
    /// Unique key identifying this effect's target (e.g., absolute path bytes).
    fn key(&self) -> Box<[u8]>;

    /// Extract the hash of the value part of this effect for comparison.
    fn value_hash(&self) -> u128;

    /// Perform the side effect, routing through the per-key state machine.
    ///
    /// Implementations typically dispatch into [`EffectStateStorage::run_apply`] with `Some(body)`
    /// when content was materialized at capture time, or `None` when capture observed
    /// `Applied { value_hash }` matching the new hash and elided content materialization.
    ///
    /// The returned error is type-erased to `Arc<dyn EffectError>` because the per-key state
    /// machine in [`EffectStateStorage::run_apply`] caches results across calls (including the
    /// dedup-hit short-circuit path) and the cached error must have a uniform type across all
    /// callers writing to the same key.
    fn apply(&self) -> impl Future<Output = Result<(), ApplyOutcome<Arc<dyn EffectError>>>> + Send;
}

/// Outcome of [`CapturedEffect::apply`]. Distinguishes a side-effect failure (terminal) from a
/// soft failure where the captured form had no content and storage state diverged between
/// capture and apply (recoverable via [`Effects::apply`]'s invalidator path).
#[derive(Debug)]
pub enum ApplyOutcome<E> {
    /// The side effect itself failed.
    Failed(E),
    /// Capture short-circuited content materialization (observed `Applied { matching }` in
    /// storage), but by apply time the storage state had diverged and we have no content to
    /// re-apply. [`Effects::apply`] should invalidate the producing operation and return
    /// [`EffectsError::Retry`].
    Retry,
}

/// The error type that an effect can return. We use `dyn std::error::Error` (instead of
/// [`anyhow::Error`] or [`SharedError`]) to encourage use of structured error types that can
/// potentially be transformed into `Issue`s.
///
/// We can't require that the returned error implements `Issue`:
/// - `Issue` uses `FileSystemPath`
/// - `turbo-tasks-fs` returns effect errors that should be transformed into `Issue`s.
/// - It logically doesn't make sense to define `Issue` in `turbo-tasks-fs`, `Issue` can't be
///   defined in a base crate either because it would form a circular crate dependency.
///
/// So instead, we leave it up to the caller to figure out how to downcast these errors themselves.
///
/// [`SharedError`]: crate::util::SharedError
pub trait EffectError: StdError + TraceRawVcs + NonLocalValue + Send + Sync + 'static {}
impl<T> EffectError for T where T: StdError + TraceRawVcs + NonLocalValue + Send + Sync + 'static {}

enum EffectLastApplied {
    Unapplied,
    InProgress {
        write_event: Event,
    },
    Applied {
        value_hash: u128,
        result: Result<(), Arc<dyn EffectError>>,
    },
}

/// Per-key entry in the effect state storage.
type EffectStateEntry = Arc<Mutex<EffectLastApplied>>;
/// Shared state storage for tracking applied effects. Stored on the filesystem implementation
/// (e.g. DiskFileSystemInner).
#[derive(Default)]
pub struct EffectStateStorage {
    effect_state: Mutex<FxHashMap<Box<[u8]>, EffectStateEntry>>,
}

impl EffectStateStorage {
    /// Returns true if the per-key state holds `Applied { value_hash == target, result: Ok(()) }`.
    ///
    /// Intended for use by [`Effect::capture`] to elide content materialization when the apply
    /// would dedup. Reading this from inside a turbo-tasks task is sound because
    /// [`Effects::apply`] re-checks at apply time and fires the producing task's invalidator on
    /// mismatch (via the [`ApplyOutcome::Retry`] / [`EffectsError::Retry`] pathway).
    pub fn matches_applied(&self, key: &[u8], target: u128) -> bool {
        let entry = self.effect_state.lock().get(key).cloned();
        let Some(entry) = entry else { return false };
        matches!(
            &*entry.lock(),
            EffectLastApplied::Applied {
                value_hash,
                result: Ok(()),
            } if *value_hash == target,
        )
    }

    /// Look up or create the per-key state entry.
    fn entry_for(&self, key: Box<[u8]>) -> EffectStateEntry {
        self.effect_state
            .lock()
            .entry(key)
            .or_insert_with(|| Arc::new(Mutex::new(EffectLastApplied::Unapplied)))
            .clone()
    }

    /// Coordinate an apply for `(key, value_hash)` against the per-key state machine.
    ///
    /// On a dedup hit (`Applied { value_hash == target, result }`), short-circuits and returns the
    /// cached result without calling `body`. Otherwise, transitions the entry to `InProgress`,
    /// awaits any in-flight apply for the same key, calls `body` to perform the side effect, then
    /// writes back the result as `Applied { value_hash, result }`.
    ///
    /// If `body` is `None` the captured effect has no materialized content (capture observed
    /// matching storage and elided materialization). In that case, a non-matching state forces a
    /// `Retry` — we have nothing to apply. The state entry is left in `Unapplied` for waiters.
    pub async fn run_apply<E, F, Fut>(
        &self,
        key: Box<[u8]>,
        value_hash: u128,
        body: Option<F>,
    ) -> Result<(), ApplyOutcome<Arc<dyn EffectError>>>
    where
        E: EffectError,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<(), E>> + Send,
    {
        let entry = self.entry_for(key);

        // If `body` panics or the future is dropped before completion, the guard's drop impl
        // resets the per-key state to `Unapplied` and notifies other waiters via the `Event` it
        // recovers from the previous `InProgress`, so they retry rather than deadlock or observe
        // a stale "panic" cache entry.
        struct EventGuard<'a> {
            entry: &'a EffectStateEntry,
        }
        impl Drop for EventGuard<'_> {
            fn drop(&mut self) {
                let prev_state = replace(&mut *self.entry.lock(), EffectLastApplied::Unapplied);
                let EffectLastApplied::InProgress { write_event } = prev_state else {
                    unreachable!("EventGuard: prev_state must be InProgress");
                };
                write_event.notify(usize::MAX);
            }
        }

        let begin_in_progress = |mut last_applied_guard: MutexGuard<'_, _>| {
            *last_applied_guard = EffectLastApplied::InProgress {
                write_event: Event::new(|| || "effect application in progress".to_string()),
            };
            EventGuard { entry: &entry }
        };

        let event_guard = loop {
            let listener;
            {
                let last_applied_guard = entry.lock();
                match &*last_applied_guard {
                    EffectLastApplied::Unapplied => {
                        break begin_in_progress(last_applied_guard);
                    }
                    EffectLastApplied::Applied {
                        value_hash: stored,
                        result,
                    } => {
                        if value_hash == *stored {
                            return result.clone().map_err(ApplyOutcome::Failed);
                        } else {
                            break begin_in_progress(last_applied_guard);
                        }
                    }
                    EffectLastApplied::InProgress { write_event } => {
                        // Event::listen registers the listener immediately, so notifications
                        // fired after we drop last_applied_guard cannot be missed.
                        listener = write_event.listen();
                    }
                }
            };
            listener.await;
        };

        // We hold the InProgress guard. Either run the body, or — if we have no content to
        // apply — release the guard (resetting state to Unapplied + waking waiters) and Retry.
        let Some(body) = body else {
            drop(event_guard);
            return Err(ApplyOutcome::Retry);
        };

        // Erase the body's concrete error type to `Arc<dyn EffectError>` so the cached result
        // type is uniform across all callers of the same key.
        let effect_result: Result<(), Arc<dyn EffectError>> = body()
            .await
            .map_err(|err| Arc::new(err) as Arc<dyn EffectError>);

        let prev_state = replace(
            &mut *entry.lock(),
            EffectLastApplied::Applied {
                value_hash,
                result: effect_result.clone(),
            },
        );
        forget(event_guard);

        let EffectLastApplied::InProgress { write_event } = prev_state else {
            unreachable!("Effect applied: prev_state must be InProgress");
        };
        write_event.notify(usize::MAX);

        effect_result.map_err(ApplyOutcome::Failed)
    }
}

// Private dyn-dispatch wrapper for emit-time `Effect`. Held inside `EffectInstance` cells.
// Provides only `dyn_capture` — Vc-reading capture step that runs during `take_effects`.
trait DynEffect: TraceRawVcs + NonLocalValue + Send + Sync + 'static {
    fn dyn_capture<'a>(&'a self) -> DynCaptureFuture<'a>;
}

type DynCaptureFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Box<dyn DynCapturedEffect>>> + Send + 'a>>;

impl<T> DynEffect for T
where
    T: Effect,
{
    fn dyn_capture<'a>(&'a self) -> DynCaptureFuture<'a> {
        Box::pin(async move {
            let captured = Effect::capture(self).await?;
            Ok(Box::new(captured) as Box<dyn DynCapturedEffect>)
        })
    }
}

// Private dyn-dispatch wrapper for post-capture `CapturedEffect`. Held inside `Effects.captured`
// (Vec drops on successful apply). No Vc reads. Mirrors the dynosaur pattern of
// https://github.com/spastorino/dynosaur.
pub(crate) trait DynCapturedEffect:
    TraceRawVcs + NonLocalValue + Send + Sync + 'static
{
    fn key(&self) -> Box<[u8]>;
    fn value_hash(&self) -> u128;
    fn dyn_apply<'a>(&'a self) -> DynEffectApplyFuture<'a>;
}

impl<T> DynCapturedEffect for T
where
    T: CapturedEffect,
{
    fn key(&self) -> Box<[u8]> {
        CapturedEffect::key(self)
    }

    fn value_hash(&self) -> u128 {
        CapturedEffect::value_hash(self)
    }

    fn dyn_apply<'a>(&'a self) -> DynEffectApplyFuture<'a> {
        Box::pin(async move { CapturedEffect::apply(self).await })
    }
}

type DynEffectApplyFuture<'a> =
    Pin<Box<dyn Future<Output = Result<(), ApplyOutcome<Arc<dyn EffectError>>>> + Send + 'a>>;

/// A trait to emit a task effect as collectible. This trait only has one implementation,
/// `EffectInstance` and no other implementation is allowed. The trait is private to this module so
/// that no other implementation can be added.
#[turbo_tasks::value_trait]
trait EffectCollectible {}

/// The Effect instance collectible that is emitted for effects.
#[turbo_tasks::value(serialization = "skip", cell = "new", eq = "manual")]
struct EffectInstance {
    #[turbo_tasks(debug_ignore)]
    inner: Box<dyn DynEffect>,
}

impl EffectInstance {
    fn new(effect: impl Effect) -> Self {
        Self {
            inner: Box::new(effect) as Box<dyn DynEffect>,
        }
    }
}

#[turbo_tasks::value_impl]
impl EffectCollectible for EffectInstance {}

/// Emits an effect to be applied. The effect is executed once [`Effects::apply`] is called (see
/// [`take_effects`]).
///
/// The effect will only executed once. The effect is executed outside of the current task
/// and can't read any Vcs. These need to be read before. ReadRefs can be passed into the effect.
///
/// Effects are executed in parallel, so they might need to use async locking to avoid problems.
/// Order of execution of multiple effects is not defined. You must not use multiple conflicting
/// effects to avoid non-deterministic behavior.
pub fn emit_effect(effect: impl Effect) {
    emit::<Box<dyn EffectCollectible>>(ResolvedVc::upcast(
        EffectInstance::new(effect).resolved_cell(),
    ));
}

/// Capture effects. Call this from within a [turbo-tasks operation][crate::OperationVc].
///
/// Collectibles are read from `ResolvedVc`s, so this function, and the return value of this
/// function should be applied with [`Effects::apply`].
///
/// It's important to wrap calls to this function in an [operation with a strongly consistent
/// read][crate::OperationVc::read_strongly_consistent] before applying the effects outside of the
/// operation at the top-level (e.g. in a `run_once` closure) with [`Effects::apply`].
///
/// # Example
///
/// ```rust
/// # #![feature(arbitrary_self_types_pointers)]
/// #
/// # use anyhow::Result;
/// # use turbo_tasks::{Effects, ReadRef, Vc, run_once, take_effects};
/// #
/// # async fn _wrapper() -> Result<()> {
/// # type Example = ();
/// # type Args = ();
/// # let args = ();
/// # #[turbo_tasks::function(operation)]
/// # fn some_turbo_tasks_operation(_args: Args) {}
/// #
/// #[turbo_tasks::value(serialization = "skip")]
/// struct OutputWithEffects {
///     output: ReadRef<Example>,
///     effects: Effects,
/// }
///
/// // ensure the return value and the collectibles match by using a single operation for both
/// #[turbo_tasks::function(operation)]
/// async fn some_turbo_tasks_operation_with_effects(args: Args) -> Result<Vc<OutputWithEffects>> {
///     let operation = some_turbo_tasks_operation(args);
///     // we must first read the operation to populate the collectibles
///     let output = operation.connect().await?;
///     // read the effects from the collectibles
///     let effects = take_effects(operation).await?;
///     Ok(OutputWithEffects { output, effects }.cell())
/// }
///
/// // every operation must be read with strong consistency at the top-level
/// let result_with_effects = some_turbo_tasks_operation_with_effects(args)
///     .read_strongly_consistent()
///     .await?;
///
/// // apply the effects once outside of a turbo_tasks::function at the top-level (e.g. `run_once`)
/// result_with_effects.effects.apply().await?;
/// # Ok(())
/// # }
/// ```
pub async fn take_effects(source: impl CollectiblesSource) -> Result<Effects> {
    debug_assert_not_in_top_level_task("take_effects");
    let effect_refs = source
        .take_collectibles::<Box<dyn EffectCollectible>>()
        .into_iter()
        .map(|effect| {
            if let Some(effect) = ResolvedVc::try_downcast_type::<EffectInstance>(effect) {
                effect
            } else {
                unreachable!("EffectCollectible must only be implemented by EffectInstance");
            }
        })
        .try_join()
        .await?;

    // Capture step: resolve any Vc reads now while we're still inside the producing task's
    // context. The `ReadRef<EffectInstance>`s drop at the end of this iteration, so the only
    // long-lived strong counts onto `EffectInstance` cells come from `dyn_capture` borrows
    // (transient).
    let captured: Vec<Box<dyn DynCapturedEffect>> = effect_refs
        .iter()
        .map(|effect_ref| effect_ref.inner.dyn_capture())
        .try_join()
        .await?;
    drop(effect_refs);

    // Eager per-key conflict detection. This only inspects the captured effects themselves
    // (their `key()` and `value_hash()`), so it is safe to run inside the producing task.
    // Looking up `EffectStateStorage` entries is deferred to `apply()` because that storage is
    // shared mutable state that other top-level applies can mutate concurrently.
    let unique_keys = build_unique_keys(&captured);

    // Grab an invalidator on the *producing task*. Used only on the retry path: if a later
    // `apply()` call discovers the captured Vec was dropped and the per-key state machine no
    // longer carries our `Applied { value_hash }`, we invalidate this task so the producer
    // re-runs and emits fresh effects.
    let invalidator = get_invalidator()
        .expect("take_effects must be called from within a turbo-tasks task context");

    Ok(Effects::new(captured, unique_keys, invalidator))
}

#[derive(thiserror::Error, Debug, TraceRawVcs, NonLocalValue)]
#[error("Conflicting effects for the same key (key length: {key_len} bytes)")]
struct ConflictingEffectError {
    key_len: usize,
}

/// Error returned by [`Effects::apply`]. Callers should retry on `Retry`; everything else is
/// terminal.
#[derive(thiserror::Error, Debug, Clone)]
pub enum EffectsError {
    /// A side effect failed during apply. Holds the first error encountered.
    #[error(transparent)]
    Apply(Arc<dyn EffectError>),

    /// Two effects emitted the same key with different value_hashes; no apply happened.
    #[error("conflicting effects for the same key (key length: {0} bytes)")]
    Conflict(usize),

    /// The captured effects were dropped after a previous successful apply, and the shared
    /// per-key state for at least one effect no longer carries a matching `Applied { value_hash
    /// }`. The producing operation has been invalidated; the caller should re-read the operation
    /// and call `apply()` again on the fresh [`Effects`] value.
    #[error("effect state was reset; producing operation has been invalidated, retry required")]
    Retry,
}

impl From<Arc<dyn EffectError>> for EffectsError {
    fn from(err: Arc<dyn EffectError>) -> Self {
        EffectsError::Apply(err)
    }
}

/// Dedup'd indices into the captured Vec — one entry per unique key. Computed eagerly in
/// [`take_effects`] purely from the captured effects (no [`EffectStateStorage`] interaction);
/// the apply-side state machine in [`EffectStateStorage::run_apply`] handles per-key hash dedup.
type UniqueKeys = Result<Vec<usize>, Arc<ConflictingEffectError>>;

/// Slice of captured effects, individually Arc'd. Each effect is `Arc<dyn DynCapturedEffect>`
/// so callers can cheaply clone a Send handle out across `.await` boundaries without holding
/// the outer mutex.
type CapturedSlice = Arc<[Arc<dyn DynCapturedEffect>]>;

/// Captured effects from an operation. This struct can be used to return Effects from a turbo-tasks
/// function and apply them later.
///
/// # Cell semantics
///
/// `Effects` uses `cell = "new"`: every producer re-execution allocates a fresh cell value and
/// the prior cell is dropped. Cell-level dedup of `Effects` is given up; per-key dedup at apply
/// time is provided by [`EffectStateStorage`]'s state machine (see
/// [`EffectStateStorage::run_apply`]), which short-circuits when storage already holds
/// `Applied { value_hash }` matching the new hash.
///
/// `Effects::apply` is idempotent and safe to call multiple times on the same value — the state
/// machine in `run_apply` ensures each underlying side effect runs at most once per stored
/// `(key, value_hash)` pair across all callers.
#[turbo_tasks::value(shared, eq = "manual", serialization = "skip", cell = "new")]
pub struct Effects {
    /// Pre-resolved effects awaiting application. Lives for the lifetime of the cell — released
    /// when the producer reruns and `cell = "new"` overwrites the cell, which is when any
    /// upstream `ReadRef` strong-count cascades are naturally released.
    #[turbo_tasks(debug_ignore, trace_ignore)]
    captured: CapturedSlice,
    /// Captured at `take_effects` time. `None` for `Effects::empty()` (nothing to retry).
    #[turbo_tasks(debug_ignore, trace_ignore)]
    invalidator: Option<Invalidator>,
    /// Unique key info computed eagerly in `take_effects`. Holds the dedup'd `(idx, value_hash)`
    /// per unique key, or a `ConflictingEffectError` if two captured effects share a key with
    /// different hashes. No [`EffectStateStorage`] interaction here — that is deferred to
    /// `apply()`.
    #[turbo_tasks(debug_ignore, trace_ignore)]
    unique_keys: Arc<UniqueKeys>,
}

/// `PartialEq`/`Eq` are compat shims so containing structs (which derive `PartialEq`/`Eq` via
/// `turbo_tasks::value`) can still embed `Effects`. The actual cell-update strategy for `Effects`
/// itself is `cell = "new"` — see the doc-comment above — so this `PartialEq` is not consulted
/// for `Effects` cells. We always return `false` to match `cell = "new"` semantics for the
/// wrapper structs (they should also refresh on every producer run).
impl PartialEq for Effects {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}
impl Eq for Effects {}

impl Effects {
    /// An `Effects` value with no effects. Used by callers that need a placeholder where no
    /// side effects were collected.
    #[cfg(test)]
    fn empty() -> Self {
        Self {
            captured: Arc::from(Vec::new()),
            invalidator: None,
            unique_keys: Arc::new(Ok(Vec::new())),
        }
    }

    fn new(
        captured: Vec<Box<dyn DynCapturedEffect>>,
        unique_keys: UniqueKeys,
        invalidator: Invalidator,
    ) -> Self {
        // Convert Box<dyn> into Arc<dyn> per slot. Each Arc is independently Send/Sync.
        let captured: CapturedSlice = captured
            .into_iter()
            .map(Arc::<dyn DynCapturedEffect>::from)
            .collect();
        Self {
            captured,
            invalidator: Some(invalidator),
            unique_keys: Arc::new(unique_keys),
        }
    }

    /// Applies all effects that have been captured.
    ///
    /// Dispatch goes through each captured effect's [`CapturedEffect::apply`] (via
    /// [`EffectStateStorage::run_apply`]) which handles the per-key state machine, dedup hits,
    /// in-progress coordination, and panic recovery. The dispatch is idempotent — calling
    /// `apply()` multiple times on the same `Effects` value runs each underlying side effect at
    /// most once per stored `(key, value_hash)` pair.
    ///
    /// If any captured effect signals [`ApplyOutcome::Retry`] (its content was elided at capture
    /// time and storage state diverged between capture and apply), the producing task is
    /// invalidated and [`EffectsError::Retry`] is returned after the remaining keys finish.
    /// Side-effect failures (`ApplyOutcome::Failed`) propagate as [`EffectsError::Apply`]; the
    /// first such error wins.
    ///
    /// `apply` must only be used in a "top-level" task (e.g. [`run_once`][crate::run_once]),
    /// after [`take_effects`] is called from an [operation read with strong
    /// consistency][crate::OperationVc::read_strongly_consistent].
    ///
    /// See [`take_effects`] for example usage.
    pub async fn apply(&self) -> Result<(), EffectsError> {
        debug_assert_in_top_level_task(
            "Effects::apply must be called from a top-level task to avoid unintended \
             re-executions due to eventual consistency",
        );
        let unique = match self.unique_keys.as_ref() {
            Ok(unique) => unique.as_slice(),
            Err(err) => return Err(EffectsError::Conflict(err.key_len)),
        };
        if unique.is_empty() {
            return Ok(());
        }

        let span = tracing::info_span!("apply effects", count = unique.len());
        let captured = &self.captured;

        async {
            // Collect a single `Retry` signal across the parallel apply so we invalidate at most
            // once at the end of the batch. `Apply` errors still take precedence — they fail-fast
            // through the `try_for_each_concurrent`.
            let needs_retry = AtomicBool::new(false);
            let result: Result<(), EffectsError> = futures::stream::iter(unique.iter())
                .map(Ok::<_, EffectsError>)
                .try_for_each_concurrent(
                    APPLY_EFFECTS_CONCURRENCY_LIMIT,
                    async |idx| match captured[*idx].dyn_apply().await {
                        Ok(()) => Ok(()),
                        Err(ApplyOutcome::Failed(err)) => Err(EffectsError::Apply(err)),
                        Err(ApplyOutcome::Retry) => {
                            needs_retry.store(true, Ordering::Relaxed);
                            Ok(())
                        }
                    },
                )
                .await;

            match result {
                Err(e) => Err(e),
                Ok(()) if needs_retry.load(Ordering::Relaxed) => self.signal_retry(),
                Ok(()) => Ok(()),
            }
        }
        .instrument(span)
        .await
    }

    /// Invalidate the producing task (if any) and return `EffectsError::Retry`. Used when some
    /// captured effect signaled [`ApplyOutcome::Retry`] (capture elided content materialization
    /// but storage state diverged before apply).
    fn signal_retry(&self) -> Result<(), EffectsError> {
        if let Some(invalidator) = self.invalidator {
            with_turbo_tasks(|tt| invalidator.invalidate(&**tt));
        }
        Err(EffectsError::Retry)
    }
}

/// Build deduped `(idx, value_hash)` info from the captured slice. Detects per-key value-hash
/// conflicts. This is the eager half of effect deduplication — it inspects only the captured
/// effects themselves (no [`EffectStateStorage`] interaction) and is therefore safe to call
/// from inside a turbo-tasks task in [`take_effects`].
fn build_unique_keys(captured: &[Box<dyn DynCapturedEffect>]) -> UniqueKeys {
    let mut by_key: FxHashMap<Box<[u8]>, usize> = FxHashMap::default();
    for (idx, effect) in captured.iter().enumerate() {
        match by_key.entry(effect.key()) {
            hash_map::Entry::Vacant(entry) => {
                entry.insert(idx);
            }
            hash_map::Entry::Occupied(entry) => {
                if captured[*entry.get()].value_hash() != effect.value_hash() {
                    return Err(Arc::new(ConflictingEffectError {
                        key_len: entry.key().len(),
                    }));
                }
            }
        }
    }

    let mut keys: Vec<usize> = by_key.into_values().collect();
    // Sort by idx so the order is deterministic — useful for stable tracing/logging.
    keys.sort_unstable();
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use crate::{CollectiblesSource, Effects, take_effects};

    #[test]
    #[allow(dead_code)]
    fn is_send() {
        fn assert_send<T: Send>(_: T) {}
        fn check_effects_apply() {
            assert_send(Effects::empty().apply());
        }
        fn check_take_effects<T: CollectiblesSource + Send + Sync>(t: T) {
            assert_send(take_effects(t));
        }
    }
}
