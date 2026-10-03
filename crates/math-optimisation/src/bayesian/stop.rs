use std::sync::atomic::{AtomicBool, Ordering};

/// Latches the first stop observation across parallel objective workers.
pub(super) struct StopCheck<'a> {
    predicate: &'a (dyn Fn() -> bool + Sync),
    stopped: AtomicBool,
}

impl<'a> StopCheck<'a> {
    pub(super) fn new(predicate: &'a (dyn Fn() -> bool + Sync)) -> Self {
        Self {
            predicate,
            stopped: AtomicBool::new(false),
        }
    }

    pub(super) fn requested(&self) -> bool {
        if self.stopped.load(Ordering::Acquire) {
            return true;
        }
        if (self.predicate)() {
            self.stopped.store(true, Ordering::Release);
            return true;
        }
        self.stopped.load(Ordering::Acquire)
    }

    pub(super) fn observed(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }
}
