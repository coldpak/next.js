//! Integration tests for `take_effects` and `Effects::apply` semantics.
//!
//! These tests use a custom `Effect` impl (`TestEffectEmit`) whose `apply`
//! increments a shared counter, so we can directly assert how many times each
//! effect's side-effect actually ran across various producer/apply scenarios:
//!
//! - Duplicate sequential `.apply()` on the same `Effects` value runs each side-effect exactly once
//!   (per-key state machine short-circuits).
//! - Re-emitting the same `(key, value_hash)` set from a re-executed producer does not re-run any
//!   side-effect (the cached `Applied { value_hash }` in `EffectStateStorage` matches).
//! - Re-emitting with a changed hash for one key re-runs only that key's side-effect; unchanged
//!   sibling keys stay short-circuited.
//! - Adding / removing effects from the emitted set behaves consistently: new keys run once,
//!   removed keys' applies do not re-fire.

#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]
#![allow(clippy::needless_return)] // tokio macro-generated code doesn't respect this

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use anyhow::Result;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use turbo_tasks::{
    CapturedEffect, Effect, EffectStateStorage, Effects, NonLocalValue, OperationValue, ReadRef,
    ResolvedVc, State, TurboTasks, Vc, emit_effect, take_effects, trace::TraceRawVcs,
};
use turbo_tasks_backend::{
    BackendOptions, NoopBackingStorage, TurboTasksBackend, noop_backing_storage,
};

// =============================================================================
// Per-test shared state
// =============================================================================

/// Per-test counters + `EffectStateStorage`. Tests assert against
/// `applies_by_key` to check how many times each effect ran.
#[derive(TraceRawVcs, NonLocalValue)]
struct Shared {
    #[turbo_tasks(trace_ignore)]
    applies_by_key: Mutex<FxHashMap<u32, u64>>,
    #[turbo_tasks(trace_ignore)]
    total_applies: AtomicU64,
    /// Shared `EffectStateStorage` returned by `CapturedEffect::state_storage`.
    #[turbo_tasks(trace_ignore)]
    state_storage: EffectStateStorage,
    /// Spec driving what the producer emits. Lives in `Shared` (not in the
    /// `TestInput` cell) so the test body can mutate it directly at top-level
    /// via the `Arc<Shared>` handle, without going through a cached operation.
    /// Mutating from inside a cached operation would make the operation
    /// non-deterministic.
    #[turbo_tasks(trace_ignore)]
    spec: State<EmitSpec>,
}

impl Shared {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            applies_by_key: Mutex::new(Default::default()),
            total_applies: AtomicU64::new(0),
            state_storage: EffectStateStorage::default(),
            spec: State::new(EmitSpec::default()),
        })
    }

    fn applies_for(&self, key: u32) -> u64 {
        self.applies_by_key.lock().get(&key).copied().unwrap_or(0)
    }

    fn total(&self) -> u64 {
        self.total_applies.load(Ordering::Relaxed)
    }
}

// =============================================================================
// Test Effect impl
// =============================================================================

#[derive(TraceRawVcs, NonLocalValue)]
struct TestEffectEmit {
    key: u32,
    value_hash: u128,
    shared: Arc<Shared>,
}

impl Effect for TestEffectEmit {
    type Captured = TestEffectCaptured;

    async fn capture(&self) -> Result<TestEffectCaptured> {
        Ok(TestEffectCaptured {
            key: self.key,
            value_hash: self.value_hash,
            shared: self.shared.clone(),
        })
    }
}

#[derive(TraceRawVcs, NonLocalValue)]
struct TestEffectCaptured {
    key: u32,
    value_hash: u128,
    shared: Arc<Shared>,
}

impl CapturedEffect for TestEffectCaptured {
    type Error = TestError;

    fn key(&self) -> Box<[u8]> {
        self.key.to_le_bytes().into()
    }

    fn value_hash(&self) -> u128 {
        self.value_hash
    }

    fn state_storage(&self) -> &EffectStateStorage {
        &self.shared.state_storage
    }

    async fn apply(&self) -> Result<(), TestError> {
        self.shared.total_applies.fetch_add(1, Ordering::Relaxed);
        *self
            .shared
            .applies_by_key
            .lock()
            .entry(self.key)
            .or_insert(0) += 1;
        Ok(())
    }
}

/// Trivial uninhabited error type. `Effect::Error` requires
/// `EffectError: StdError + TraceRawVcs + NonLocalValue + ...`, and
/// `Infallible` doesn't impl `TraceRawVcs`/`NonLocalValue`.
#[derive(Debug, thiserror::Error, TraceRawVcs, NonLocalValue)]
enum TestError {}

// =============================================================================
// Producer infrastructure
// =============================================================================

/// Spec for what the producer should emit: a list of `(key, value_hash)`
/// pairs. Stored inside `State<EmitSpec>` so mutating it invalidates the
/// producer.
///
/// `value_hash` is `u64` here (rather than `u128`) because `u128` doesn't
/// implement `TaskInput`. Widened to `u128` at emit time below.
#[derive(Clone, Default, PartialEq, Eq, Debug, TraceRawVcs, NonLocalValue, OperationValue)]
struct EmitSpec {
    pairs: Vec<(u32, u64)>,
}

/// Input cell. Just holds an `Arc<Shared>`. The `State<EmitSpec>` that
/// drives the producer lives inside `Shared` (see [`Shared::spec`]) so it
/// can be mutated from top-level outside any cached operation.
#[turbo_tasks::value(eq = "manual", serialization = "skip")]
struct TestInput {
    #[turbo_tasks(trace_ignore, debug_ignore)]
    shared: Arc<Shared>,
}

impl PartialEq for TestInput {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl TestInput {
    /// Construct a fresh `TestInput` cell. Returns the `ResolvedVc` alongside
    /// an `Arc<Shared>` clone so the test body can mutate the spec and assert
    /// against the apply counters without doing any Vc reads.
    fn new() -> (ResolvedVc<Self>, Arc<Shared>) {
        let shared = Shared::new();
        let cell = Self {
            shared: shared.clone(),
        }
        .resolved_cell();
        (cell, shared)
    }
}

#[turbo_tasks::function(operation, root)]
async fn producer_operation(input: ResolvedVc<TestInput>) -> Result<()> {
    let input = input.await?;
    let shared = input.shared.clone();
    let spec = shared.spec.get().clone();
    for (key, value_hash) in spec.pairs {
        emit_effect(TestEffectEmit {
            key,
            value_hash: value_hash as u128,
            shared: shared.clone(),
        });
    }
    Ok(())
}

/// Read the current spec from `input` and emit the corresponding effects,
/// returning the captured `Effects`. State mutation must happen OUTSIDE this
/// operation (at top-level via `set_spec`), otherwise the operation becomes
/// non-deterministic and turbo-tasks may re-run it with stale inputs.
#[turbo_tasks::function(operation, root)]
async fn extract_effects(input: ResolvedVc<TestInput>) -> Result<Vc<Effects>> {
    let producer = producer_operation(input);
    let _ = producer.resolve().strongly_consistent().await?;
    Ok(take_effects(producer).await?.cell())
}

/// Mutate the spec on the shared handle and return the resulting `Effects`.
/// State mutation is synchronous at top-level (so it doesn't make any
/// operation non-deterministic); only the `take_effects` step runs inside an
/// operation root so it gets strongly-consistent read semantics.
async fn emit_and_take(
    shared: &Shared,
    input: ResolvedVc<TestInput>,
    pairs: Vec<(u32, u64)>,
) -> Result<ReadRef<Effects>> {
    shared.spec.set(EmitSpec { pairs });
    Ok(extract_effects(input).read_strongly_consistent().await?)
}

// =============================================================================
// Test harness
// =============================================================================

fn create_tt() -> Arc<TurboTasks<TurboTasksBackend<NoopBackingStorage>>> {
    TurboTasks::new(TurboTasksBackend::new(
        BackendOptions::default(),
        noop_backing_storage(),
    ))
}

// =============================================================================
// Tests
// =============================================================================

/// A duplicate sequential `.apply()` on the same `Effects` must run each
/// underlying side-effect exactly once. The first call populates the per-key
/// `EffectStateStorage` with `Applied { value_hash }`; the second call sees
/// `Effects.captured == None` and falls through `apply_post_drop` where every
/// state entry is `Applied` with a matching hash → no `dyn_apply` invocation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn duplicate_apply_runs_once() {
    let tt = create_tt();
    tt.run_once(async move {
        let (input, shared) = TestInput::new();

        let effects = emit_and_take(&shared, input, vec![(1, 0xAAAA)]).await?;

        effects.apply().await?;
        assert_eq!(shared.applies_for(1), 1, "first apply runs the effect");

        effects.apply().await?;
        assert_eq!(
            shared.applies_for(1),
            1,
            "second apply on the same Effects must not re-run"
        );

        anyhow::Ok(())
    })
    .await
    .unwrap()
}

/// Re-running the producer (e.g. because something upstream invalidated it)
/// with the same `(key, value_hash)` set produces a fresh `Effects` value
/// whose identity multiset matches the previous one. `.apply()` on the new
/// `Effects` short-circuits through the per-key state machine because each
/// `EffectStateEntry` is still `Applied { value_hash: matching }`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reemit_unchanged_hash_does_not_reapply() {
    let tt = create_tt();
    tt.run_once(async move {
        let (input, shared) = TestInput::new();

        emit_and_take(&shared, input, vec![(1, 0xAAAA), (2, 0xBBBB)])
            .await?
            .apply()
            .await?;
        assert_eq!(shared.total(), 2, "first emit runs both effects");

        // Re-set with the same value. `State::set` is a no-op when the value
        // hasn't changed (PartialEq), but the second `extract_effects` invocation
        // is still a fresh root task — it re-takes the producer's collectibles.
        emit_and_take(&shared, input, vec![(1, 0xAAAA), (2, 0xBBBB)])
            .await?
            .apply()
            .await?;
        assert_eq!(
            shared.total(),
            2,
            "re-emit with same (key, hash) must not re-run any apply"
        );

        anyhow::Ok(())
    })
    .await
    .unwrap()
}

/// Changing one effect's hash re-runs only that effect; the sibling stays
/// short-circuited by its `Applied { value_hash: matching }` state entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hash_change_reapplies_only_changed_key() {
    let tt = create_tt();
    tt.run_once(async move {
        let (input, shared) = TestInput::new();

        emit_and_take(&shared, input, vec![(1, 0xAAAA), (2, 0xBBBB)])
            .await?
            .apply()
            .await?;
        assert_eq!(shared.applies_for(1), 1);
        assert_eq!(shared.applies_for(2), 1);

        // Change key 1's hash; leave key 2 alone.
        emit_and_take(&shared, input, vec![(1, 0xCCCC), (2, 0xBBBB)])
            .await?
            .apply()
            .await?;
        assert_eq!(
            shared.applies_for(1),
            2,
            "key 1 hash changed; apply must run again"
        );
        assert_eq!(
            shared.applies_for(2),
            1,
            "key 2 unchanged; apply must short-circuit"
        );

        anyhow::Ok(())
    })
    .await
    .unwrap()
}

/// Adding a brand-new effect to the emitted set only runs the new key's
/// apply; the pre-existing keys stay short-circuited.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn adding_effect_only_runs_new_key() {
    let tt = create_tt();
    tt.run_once(async move {
        let (input, shared) = TestInput::new();

        emit_and_take(&shared, input, vec![(1, 0xAAAA)])
            .await?
            .apply()
            .await?;
        assert_eq!(shared.total(), 1);

        // Add a new effect, key 2.
        emit_and_take(&shared, input, vec![(1, 0xAAAA), (2, 0xBBBB)])
            .await?
            .apply()
            .await?;

        assert_eq!(shared.applies_for(1), 1, "key 1 already applied");
        assert_eq!(shared.applies_for(2), 1, "key 2 newly added");
        anyhow::Ok(())
    })
    .await
    .unwrap()
}

/// Removing an effect from the emitted set must not re-run the surviving
/// effect, and the removed effect's apply must not fire again either.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn removing_effect_does_not_reapply_survivors() {
    let tt = create_tt();
    tt.run_once(async move {
        let (input, shared) = TestInput::new();

        emit_and_take(&shared, input, vec![(1, 0xAAAA), (2, 0xBBBB)])
            .await?
            .apply()
            .await?;
        assert_eq!(shared.total(), 2);

        // Drop key 2.
        emit_and_take(&shared, input, vec![(1, 0xAAAA)])
            .await?
            .apply()
            .await?;

        assert_eq!(shared.applies_for(1), 1, "key 1 stays cached");
        assert_eq!(
            shared.applies_for(2),
            1,
            "key 2 was removed; its apply must not run again"
        );

        anyhow::Ok(())
    })
    .await
    .unwrap()
}
