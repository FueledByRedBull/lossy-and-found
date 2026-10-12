use std::{
    collections::VecDeque,
    sync::{
        Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::Result;
use thiserror::Error;

static CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);

pub fn request_cancel() {
    CANCEL_REQUESTED.store(true, Ordering::SeqCst);
    work_budget().wake();
}

pub fn clear_cancel() {
    CANCEL_REQUESTED.store(false, Ordering::SeqCst);
}

pub fn cancellation_requested() -> bool {
    CANCEL_REQUESTED.load(Ordering::SeqCst)
}

/// Why waiting for a budget ended without a grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum WaitEnded {
    #[error("cancelled while waiting for the work budget")]
    Cancelled,
    #[error("timed out while waiting for the work budget")]
    TimedOut,
}

#[derive(Debug)]
pub(crate) struct BudgetState {
    limit: u64,
    used: u64,
    /// Tickets of the requests that are waiting, oldest first.
    pub(crate) queue: VecDeque<u64>,
    next_ticket: u64,
}

/// A counted budget whose requests are served in arrival order, so a large
/// request is not starved by a stream of small ones. A request larger than the
/// whole budget is clamped to it and runs alone. Nothing is held while waiting.
///
/// The rule that keeps it deadlock free is the caller's: a permit is only held
/// by work that finishes without asking for another permit. File workers and
/// anything else that waits on other work hold none.
#[derive(Debug)]
pub struct Budget {
    state: Mutex<BudgetState>,
    changed: Condvar,
}

/// Waiters re-check cancellation and their deadline at least this often, so a
/// wake-up that nobody sent cannot strand them.
const WAIT_SLICE: Duration = Duration::from_millis(25);

impl Budget {
    pub const fn new(limit: u64) -> Self {
        Self {
            state: Mutex::new(BudgetState {
                limit: if limit == 0 { 1 } else { limit },
                used: 0,
                queue: VecDeque::new(),
                next_ticket: 0,
            }),
            changed: Condvar::new(),
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, BudgetState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn wait_slice<'a>(
        &self,
        state: MutexGuard<'a, BudgetState>,
        deadline: Option<Instant>,
    ) -> MutexGuard<'a, BudgetState> {
        let slice = deadline.map_or(WAIT_SLICE, |deadline| {
            deadline
                .saturating_duration_since(Instant::now())
                .min(WAIT_SLICE)
        });
        self.changed
            .wait_timeout(state, slice)
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .0
    }

    pub fn set_limit(&self, limit: u64) {
        self.lock().limit = limit.max(1);
        self.changed.notify_all();
    }

    pub fn limit(&self) -> u64 {
        self.lock().limit
    }

    pub fn used(&self) -> u64 {
        self.lock().used
    }

    /// Makes every waiter re-check cancellation, its deadline and its condition.
    pub fn wake(&self) {
        drop(self.lock());
        self.changed.notify_all();
    }

    /// Waits for `units` (at least one, at most the whole budget), until the
    /// deadline passes, or until cancellation is requested.
    pub fn acquire(&self, units: u64, deadline: Option<Instant>) -> Result<Permit<'_>, WaitEnded> {
        self.acquire_unless(units, deadline, cancellation_requested)
    }

    pub fn acquire_unless(
        &self,
        units: u64,
        deadline: Option<Instant>,
        stop: impl Fn() -> bool,
    ) -> Result<Permit<'_>, WaitEnded> {
        let mut state = self.lock();
        let ticket = state.next_ticket;
        state.next_ticket += 1;
        state.queue.push_back(ticket);
        loop {
            let ended = if stop() {
                Some(WaitEnded::Cancelled)
            } else if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                Some(WaitEnded::TimedOut)
            } else {
                None
            };
            let need = units.clamp(1, state.limit);
            if ended.is_none()
                && state.queue.front() == Some(&ticket)
                && state.limit.saturating_sub(state.used) >= need
            {
                state.queue.pop_front();
                state.used += need;
                // Room may be left for the request behind this one.
                self.changed.notify_all();
                return Ok(Permit {
                    budget: self,
                    units: need,
                });
            }
            if let Some(ended) = ended {
                state.queue.retain(|queued| *queued != ticket);
                self.changed.notify_all();
                return Err(ended);
            }
            state = self.wait_slice(state, deadline);
        }
    }

    /// Waits until `blocked` is false, for conditions that are not a count,
    /// such as temporary files in use. `wake` makes a waiter check sooner.
    pub fn wait_while(
        &self,
        deadline: Option<Instant>,
        blocked: impl Fn() -> bool,
    ) -> Result<(), WaitEnded> {
        let mut state = self.lock();
        loop {
            if cancellation_requested() {
                return Err(WaitEnded::Cancelled);
            }
            if !blocked() {
                return Ok(());
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(WaitEnded::TimedOut);
            }
            state = self.wait_slice(state, deadline);
        }
    }
}

/// Units taken from a [`Budget`], returned when dropped.
#[derive(Debug)]
pub struct Permit<'a> {
    budget: &'a Budget,
    units: u64,
}

impl Permit<'_> {
    pub fn units(&self) -> u64 {
        self.units
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut state = self.budget.lock();
        state.used = state.used.saturating_sub(self.units);
        drop(state);
        self.budget.changed.notify_all();
    }
}

/// The budget of threads of work shared by everything this process runs:
/// external tools take as many units as the threads they are told to use,
/// in-process encode, decode and metric work the threads they use, and a GPU
/// submission one.
fn work_budget() -> &'static Budget {
    static WORK: Budget = Budget::new(u64::MAX);
    &WORK
}

pub fn configure_work_budget(units: usize) {
    work_budget().set_limit(units as u64);
}

/// Takes `units` from the work budget for one piece of work that does not
/// itself wait for another permit.
pub fn acquire_work(units: usize, deadline: Option<Instant>) -> Result<Permit<'static>, WaitEnded> {
    work_budget().acquire(units as u64, deadline)
}
