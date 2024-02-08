use std::sync::{atomic::AtomicUsize, RwLock};

pub struct MessagePipeline {
    pub current_line: AtomicUsize,
    pub errors: AtomicUsize,
    pub warnings: AtomicUsize,
    pub other: AtomicUsize,
    pub file_path: &'static str,
    pub project_name: &'static str,
}

#[derive(Debug)]
pub enum Message {
    Error { msg: String, notes: Vec<String> },
    Warning { msg: String, notes: Vec<String> },
    Info { msg: String, notes: Vec<String> },
    Raw { msg: String, notes: Vec<String> },
    Panic { msg: String, notes: Vec<String> },
    _Unimplemented { msg: String, notes: Vec<String> },
}

#[inline(always)]
pub(self) fn display_message(
    _pipeline: &MessagePipeline,
    kinda: colored::ColoredString,
    msg: String,
    notes: Vec<String>,
) {
    use colored::Colorize as _;
    eprintln!(
        "{}: {}\n {} {}:{}\n{}",
        kinda,
        msg.white().bold(),
        "-->".blue(),
        _pipeline.file_path,
        match _pipeline
            .current_line
            .load(std::sync::atomic::Ordering::Relaxed)
            .saturating_sub(1)
        {
            0 => 1,
            n => n,
        },
        notes.join("\n")
    );
}

impl MessagePipeline {
    #[track_caller]
    pub fn process_message(&self, msg: Message) {
        use colored::Colorize as _;
        dbg!(core::panic::Location::caller());
        match msg {
            Message::Error { msg, notes } => {
                display_message(self, "error".red().bold(), msg, notes);
                self.errors
                    .fetch_add(1, std::sync::atomic::Ordering::Release);
            }
            Message::Warning { msg, notes } => {
                display_message(self, "warning".yellow().bold(), msg, notes);
                self.warnings
                    .fetch_add(1, std::sync::atomic::Ordering::Release);
            }
            Message::Info { msg, notes } => {
                display_message(self, "info".cyan().bold(), msg, notes);
                self.other
                    .fetch_add(1, std::sync::atomic::Ordering::Release);
            }
            Message::Raw { msg, notes } => match msg.as_str() {
                "user" => println!("[Pipeline::Raw(user)] {}", notes.join("\n")),
                _ => eprintln!("[Pipeline::Raw] {}", notes.join("\n")),
            },
            Message::Panic { msg, notes } => {
                panic!("[Pipeline::Panic] {}{}", msg, notes.join("\n"))
            }
            _ => panic!("failed to process message in pipeline"),
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
