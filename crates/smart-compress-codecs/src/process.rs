use std::{
    ffi::OsString,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::OnceLock,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use thiserror::Error;

use crate::budget::{Permit, WaitEnded, acquire_work, cancellation_requested};

pub(crate) const MAX_PROCESS_OUTPUT_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessFailureKind {
    Start,
    Poll,
    Timeout,
    Cancelled,
    Exit,
    Output,
}

#[derive(Debug, Error)]
#[error("{} {kind:?} after {elapsed_ms} ms{exit_suffix}: {message}", .executable.file_name().unwrap_or_default().to_string_lossy())]
pub struct ProcessFailure {
    pub executable: PathBuf,
    pub kind: ProcessFailureKind,
    pub exit_code: Option<i32>,
    pub message: String,
    pub elapsed_ms: u64,
    pub termination_requested: bool,
    exit_suffix: String,
}

impl ProcessFailure {
    pub(crate) fn new(
        executable: &Path,
        kind: ProcessFailureKind,
        exit_code: Option<i32>,
        message: String,
        elapsed_ms: u64,
        termination_requested: bool,
    ) -> Self {
        let exit_suffix = exit_code.map_or(String::new(), |code| format!(" (exit {code})"));
        Self {
            executable: executable.to_path_buf(),
            kind,
            exit_code,
            message,
            elapsed_ms,
            termination_requested,
            exit_suffix,
        }
    }
}

#[derive(Debug)]
pub(crate) struct CapturedStream {
    pub(crate) bytes: Vec<u8>,
    pub(crate) truncated: bool,
}

#[derive(Debug, Clone)]
pub struct ProcessOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub elapsed_ms: u64,
}

pub(crate) fn run_command(
    executable: &Path,
    args: &[OsString],
    timeout: Duration,
    threads: usize,
) -> Result<()> {
    run_process_threads(executable, args, timeout, threads).map(|_| ())
}

/// One unit of the work budget for an in-process step that runs on a single thread.
pub(crate) fn work_unit(timeout: Duration) -> Result<Permit<'static>> {
    Ok(acquire_work(1, Some(Instant::now() + timeout))?)
}

/// Runs a single-threaded tool.
pub fn run_process(
    executable: &Path,
    args: &[OsString],
    timeout: Duration,
) -> Result<ProcessOutput> {
    run_process_threads(executable, args, timeout, 1)
}

/// Runs a tool that has been told to use `threads` threads, which is what it
/// takes from the work budget. The wait for the budget is bounded by the same
/// timeout as the run.
pub fn run_process_threads(
    executable: &Path,
    args: &[OsString],
    timeout: Duration,
    threads: usize,
) -> Result<ProcessOutput> {
    let waiting = Instant::now();
    let _permit = acquire_work(threads, Some(waiting + timeout)).map_err(|ended| {
        anyhow!(ProcessFailure::new(
            executable,
            match ended {
                WaitEnded::Cancelled => ProcessFailureKind::Cancelled,
                WaitEnded::TimedOut => ProcessFailureKind::Timeout,
            },
            None,
            ended.to_string(),
            waiting.elapsed().as_millis() as u64,
            true,
        ))
    })?;
    let start = Instant::now();
    let mut command = Command::new(executable);
    command.args(args);
    // Tools get their own folder as working directory, not the caller's, so
    // a DLL they fail to find beside themselves is not picked up from there.
    // Every path in `args` is therefore absolute.
    if let Some(dir) = executable
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
    {
        command.current_dir(dir);
    }
    let mut child = quiet(&mut command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            anyhow!(ProcessFailure::new(
                executable,
                ProcessFailureKind::Start,
                None,
                error.to_string(),
                start.elapsed().as_millis() as u64,
                false,
            ))
        })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("capturing stdout from {}", executable.display()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("capturing stderr from {}", executable.display()))?;
    let stdout_reader = spawn_capped_reader(stdout);
    let stderr_reader = spawn_capped_reader(stderr);

    let (status, timed_out) = loop {
        if cancellation_requested() {
            terminate_process_tree(child.id());
            let _ = child.kill();
            let status = child
                .wait()
                .with_context(|| format!("waiting for cancelled {}", executable.display()))?;
            let stdout = join_capped_reader(stdout_reader)
                .with_context(|| format!("reading stdout from {}", executable.display()))?;
            let stderr = join_capped_reader(stderr_reader)
                .with_context(|| format!("reading stderr from {}", executable.display()))?;
            return Err(anyhow!(ProcessFailure::new(
                executable,
                ProcessFailureKind::Cancelled,
                status.code(),
                useful_output(&stderr, &stdout),
                start.elapsed().as_millis() as u64,
                true,
            )));
        }
        match child.try_wait() {
            Ok(Some(status)) => break (status, false),
            // Poll quickly at first: on small images most tools exit within
            // a few milliseconds and a fixed wait would dominate.
            Ok(None) if start.elapsed() <= timeout => thread::sleep(Duration::from_millis(
                (1 + start.elapsed().as_millis() as u64 / 10).min(20),
            )),
            Ok(None) => {
                terminate_process_tree(child.id());
                let _ = child.kill();
                let status = child
                    .wait()
                    .with_context(|| format!("waiting for timed-out {}", executable.display()))?;
                break (status, true);
            }
            Err(error) => {
                terminate_process_tree(child.id());
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_capped_reader(stdout_reader);
                let _ = join_capped_reader(stderr_reader);
                return Err(anyhow!(ProcessFailure::new(
                    executable,
                    ProcessFailureKind::Poll,
                    None,
                    error.to_string(),
                    start.elapsed().as_millis() as u64,
                    true,
                )));
            }
        }
    };

    let stdout = join_capped_reader(stdout_reader)
        .with_context(|| format!("reading stdout from {}", executable.display()))?;
    let stderr = join_capped_reader(stderr_reader)
        .with_context(|| format!("reading stderr from {}", executable.display()))?;

    if timed_out {
        return Err(anyhow!(ProcessFailure::new(
            executable,
            ProcessFailureKind::Timeout,
            status.code(),
            useful_output(&stderr, &stdout),
            start.elapsed().as_millis() as u64,
            true,
        )));
    }
    if status.success() {
        return Ok(ProcessOutput {
            stdout: stdout.bytes,
            stderr: stderr.bytes,
            stdout_truncated: stdout.truncated,
            stderr_truncated: stderr.truncated,
            elapsed_ms: start.elapsed().as_millis() as u64,
        });
    }

    Err(anyhow!(ProcessFailure::new(
        executable,
        ProcessFailureKind::Exit,
        status.code(),
        useful_output(&stderr, &stdout),
        start.elapsed().as_millis() as u64,
        false,
    )))
}

/// Keeps console tools from opening a window of their own when the caller
/// has no console, as the windowed control panel does. Their output is piped
/// either way.
pub fn quiet(command: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

/// A program of the Windows folder by absolute path, so a file of the same
/// name in the working directory or on a `PATH` entry is never started. Only
/// meaningful on Windows, the only place it is called.
pub fn system_program(relative: &str) -> PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        std::env::var_os("SystemRoot")
            .map(PathBuf::from)
            .filter(|root| root.is_absolute())
            .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
    })
    .join(relative)
}

#[cfg(windows)]
fn terminate_process_tree(process_id: u32) {
    let _ = quiet(
        Command::new(system_program(r"System32\taskkill.exe")).args([
            "/PID",
            &process_id.to_string(),
            "/T",
            "/F",
        ]),
    )
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .status();
}

#[cfg(not(windows))]
fn terminate_process_tree(_process_id: u32) {}

fn spawn_capped_reader<R>(reader: R) -> JoinHandle<std::io::Result<CapturedStream>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || read_capped(reader, MAX_PROCESS_OUTPUT_BYTES))
}

fn join_capped_reader(
    reader: JoinHandle<std::io::Result<CapturedStream>>,
) -> Result<CapturedStream> {
    reader
        .join()
        .map_err(|_| anyhow!("process output reader panicked"))?
        .map_err(anyhow::Error::from)
}

pub(crate) fn read_capped(mut reader: impl Read, limit: usize) -> std::io::Result<CapturedStream> {
    let mut bytes = Vec::with_capacity(limit.min(8 * 1024));
    let mut buffer = [0_u8; 8 * 1024];
    let mut truncated = false;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let remaining = limit.saturating_sub(bytes.len());
        let retained = read.min(remaining);
        bytes.extend_from_slice(&buffer[..retained]);
        truncated |= retained < read;
    }
    Ok(CapturedStream { bytes, truncated })
}

pub(crate) fn output_suffix(stderr: &CapturedStream, stdout: &CapturedStream) -> String {
    let (message, truncated) = first_output_line(stderr)
        .map(|message| (message, stderr.truncated))
        .or_else(|| first_output_line(stdout).map(|message| (message, stdout.truncated)))
        .unwrap_or_else(|| {
            (
                "process failed".to_string(),
                stderr.truncated || stdout.truncated,
            )
        });
    if truncated {
        format!(": {message} [output truncated]")
    } else {
        format!(": {message}")
    }
}

fn useful_output(stderr: &CapturedStream, stdout: &CapturedStream) -> String {
    output_suffix(stderr, stdout)
        .strip_prefix(": ")
        .unwrap_or("process failed")
        .to_string()
}

fn first_output_line(output: &CapturedStream) -> Option<String> {
    String::from_utf8_lossy(&output.bytes)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string)
}
