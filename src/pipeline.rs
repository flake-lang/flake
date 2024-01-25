use std::sync::{atomic::AtomicUsize, RwLock};

pub struct MessagePipeline {
    pub current_line: AtomicUsize,
    pub errors: AtomicUsize,
    pub warnings: AtomicUsize,
    pub other: AtomicUsize,
    pub file_path: &'static str,
    pub project_name: &'static str,
}

pub enum Message {
    Error { err: String, notes: Vec<String> },
}

impl MessagePipeline {
    pub fn process_message(&self, msg: Message) {
        match msg {
            Message::Error { err, notes } => {
                eprintln!(
                    "error: {}\n-> {}:{}\n{}",
                    err,
                    self.file_path,
                    self.current_line.load(std::sync::atomic::Ordering::Relaxed),
                    notes.join("\n")
                );
                self.errors
                    .fetch_add(1, std::sync::atomic::Ordering::Release);
            }
        }
    }

    pub fn needs_terminate(&self) -> bool {
        self.errors.load(std::sync::atomic::Ordering::Relaxed) != 0
    }
}

pub static COMPILER_PIPELINE: RwLock<MessagePipeline> = RwLock::new(MessagePipeline {
    current_line: AtomicUsize::new(0),
    errors: AtomicUsize::new(0),
    warnings: AtomicUsize::new(0),
    other: AtomicUsize::new(0),
    file_path: "test.fl",
    project_name: "test",
});
