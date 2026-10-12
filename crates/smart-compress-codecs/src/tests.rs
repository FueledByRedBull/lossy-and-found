use std::{
    ffi::OsString,
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, atomic::Ordering},
    thread,
    time::{Duration, Instant},
};

use super::*;
use super::{
    process::{CapturedStream, MAX_PROCESS_OUTPUT_BYTES, output_suffix, read_capped},
    tools::{REQUIRE_TOOLS_ENV, absolute_path_dirs, tool_dirs_for, tools_or_skip_with},
};

#[test]
fn a_test_without_its_tools_skips_or_fails_when_tools_are_required() {
    let mut tools = ToolInventory::detect();
    tools.cwebp.enabled = true;
    tools.cjxl.enabled = false;
    tools.djxl.enabled = false;
    assert!(tools_or_skip_with(tools.clone(), &["cwebp"], true).is_some());
    assert!(tools_or_skip_with(tools.clone(), &[], true).is_some());
    assert!(tools_or_skip_with(tools.clone(), &["cwebp", "cjxl"], false).is_none());
    let failure = std::panic::catch_unwind(|| {
        let mut tools = ToolInventory::detect();
        tools.cjxl.enabled = false;
        tools.djxl.enabled = false;
        tools_or_skip_with(tools, &["cjxl", "djxl"], true)
    })
    .expect_err("a missing tool must fail the test when tools are required");
    let message = failure
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("missing cjxl, djxl"), "{message}");
    assert!(message.contains(REQUIRE_TOOLS_ENV), "{message}");
    assert_eq!(
        tool_names_for(&[Codec::Png, Codec::Webp, Codec::Avif, Codec::Jxl]),
        ["cwebp", "avifenc", "avifdec", "cjxl", "djxl"]
    );
}

#[test]
fn a_format_needs_its_decoder_as_well_as_its_encoder() {
    let mut tools = ToolInventory::detect();
    for tool in [
        &mut tools.cwebp,
        &mut tools.avifenc,
        &mut tools.avifdec,
        &mut tools.cjxl,
        &mut tools.djxl,
    ] {
        tool.enabled = true;
    }
    assert!(
        [
            Codec::Png,
            Codec::Jpeg,
            Codec::Webp,
            Codec::Avif,
            Codec::Jxl
        ]
        .into_iter()
        .all(|codec| tools.available_for(codec))
    );

    tools.avifdec.enabled = false;
    assert!(!tools.available_for(Codec::Avif));
    assert_eq!(tools.missing_for(Codec::Avif), ["avifdec"]);
    tools.cjxl.enabled = false;
    tools.djxl.enabled = false;
    assert_eq!(tools.missing_for(Codec::Jxl), ["cjxl", "djxl"]);
    // The built-in formats and WebP, which is read back in process, are untouched.
    assert!(tools.available_for(Codec::Png) && tools.available_for(Codec::Jpeg));
    assert!(tools.available_for(Codec::Webp));
}

#[cfg(windows)]
#[test]
fn system_programs_are_started_by_absolute_path() {
    for relative in [
        r"System32\taskkill.exe",
        r"System32\cmd.exe",
        "explorer.exe",
    ] {
        let path = system_program(relative);
        assert!(path.is_absolute() && path.is_file(), "{}", path.display());
    }
}

#[test]
fn a_process_failure_names_the_executable_without_its_folder() {
    let failure = ProcessFailure::new(
        Path::new("/home/someone/tools/cwebp.exe"),
        ProcessFailureKind::Exit,
        Some(1),
        "bad input".to_string(),
        5,
        false,
    );
    let text = failure.to_string();
    assert!(text.starts_with("cwebp.exe Exit"), "{text}");
    assert!(!text.contains("someone"));
    // The full path stays available to the code that runs the tool.
    assert!(failure.executable.starts_with("/home/someone"));
}

#[test]
fn path_lookup_skips_empty_and_relative_entries() {
    let absolute = std::env::temp_dir();
    let joined = std::env::join_paths([
        PathBuf::new(),
        PathBuf::from("."),
        PathBuf::from("tools"),
        absolute.clone(),
    ])
    .unwrap();

    assert_eq!(absolute_path_dirs(&joined), vec![absolute]);
    // A trailing separator adds an empty entry.
    let mut trailing = std::env::temp_dir().into_os_string();
    trailing.push(if cfg!(windows) { ";" } else { ":" });
    assert_eq!(absolute_path_dirs(&trailing).len(), 1);
}

#[test]
fn tool_dirs_stay_beside_the_executable_or_its_workspace() {
    let root = tempfile::tempdir().unwrap();
    let make = |path: &str| {
        let dir = root.path().join(path);
        fs::create_dir_all(&dir).unwrap();
        dir
    };

    // A release folder: tools beside the executable. A `tools` folder
    // further up must not be used.
    make("tools");
    let release = make("downloads/smart-compress/tools/libjxl/bin");
    let release_exe_dir = release.ancestors().nth(3).unwrap().to_path_buf();
    assert_eq!(
        tool_dirs_for(&release_exe_dir),
        vec![
            release_exe_dir.join("tools"),
            release_exe_dir.join("tools/libjxl/bin")
        ]
    );

    // A development build: tools in the workspace that holds `target`.
    make("work/tools/libavif");
    let workspace = root.path().join("work");
    fs::write(workspace.join("Cargo.toml"), "").unwrap();
    let target = make("work/target/release");
    assert_eq!(
        tool_dirs_for(&target),
        vec![workspace.join("tools"), workspace.join("tools/libavif")]
    );
    // The same workspace does not apply to an executable outside `target`.
    assert!(tool_dirs_for(&workspace.join("tools/libavif")).is_empty());
}

#[test]
fn jpeg_does_not_support_alpha() {
    assert!(!Codec::Jpeg.supports_alpha());
    assert!(Codec::Webp.supports_alpha());
}

#[test]
fn quality_setting_label_is_stable() {
    assert_eq!(
        EncoderSettings::Quality {
            quality: 82,
            effort: 4
        }
        .label(),
        "q82/e4"
    );
}

#[test]
fn process_output_capture_drains_but_caps_retained_bytes() {
    let input = vec![b'x'; MAX_PROCESS_OUTPUT_BYTES + 8192];
    let captured = read_capped(Cursor::new(input), MAX_PROCESS_OUTPUT_BYTES).unwrap();

    assert_eq!(captured.bytes.len(), MAX_PROCESS_OUTPUT_BYTES);
    assert!(captured.truncated);
}

#[test]
fn process_error_prefers_first_non_empty_stderr_line() {
    let stderr = CapturedStream {
        bytes: b"\nencoder failed\nmore detail".to_vec(),
        truncated: false,
    };
    let stdout = CapturedStream {
        bytes: b"progress".to_vec(),
        truncated: false,
    };

    assert_eq!(output_suffix(&stderr, &stdout), ": encoder failed");
}

#[test]
fn process_output_exposes_capture_metadata() {
    let output = ProcessOutput {
        stdout: b"score".to_vec(),
        stderr: Vec::new(),
        stdout_truncated: false,
        stderr_truncated: false,
        elapsed_ms: 7,
    };

    assert_eq!(output.stdout, b"score");
    assert!(!output.stdout_truncated);
    assert_eq!(output.elapsed_ms, 7);
}

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
    mpsc,
};

/// Runs `scenario` on its own thread and fails, instead of hanging, if it
/// does not finish: a deadlock has to be a test failure.
fn finishes_within(limit: Duration, scenario: impl FnOnce() + Send + 'static) {
    let (done, finished) = mpsc::channel();
    thread::spawn(move || {
        scenario();
        let _ = done.send(());
    });
    finished
        .recv_timeout(limit)
        .expect("the scenario deadlocked or panicked");
}

/// Small deterministic generator so the load is the same on every run.
fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn a_budget_never_hands_out_more_than_its_size() {
    const LIMIT: u64 = 8;
    let budget = Arc::new(Budget::new(LIMIT));
    let in_use = Arc::new(AtomicU64::new(0));
    let peak = Arc::new(AtomicU64::new(0));
    finishes_within(Duration::from_secs(60), {
        let (budget, in_use, peak) = (budget.clone(), in_use.clone(), peak.clone());
        move || {
            let workers = (0..12_u64)
                .map(|seed| {
                    let (budget, in_use, peak) = (budget.clone(), in_use.clone(), peak.clone());
                    thread::spawn(move || {
                        let mut state = 0x9e37_79b9_7f4a_7c15 ^ (seed + 1);
                        for _ in 0..200 {
                            // Requests up to larger than the whole budget.
                            let units = next_random(&mut state) % 12 + 1;
                            let permit = budget.acquire_unless(units, None, || false).unwrap();
                            let now =
                                in_use.fetch_add(permit.units(), Ordering::SeqCst) + permit.units();
                            peak.fetch_max(now, Ordering::SeqCst);
                            assert!(budget.used() <= LIMIT);
                            thread::yield_now();
                            in_use.fetch_sub(permit.units(), Ordering::SeqCst);
                        }
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        }
    });
    assert!(peak.load(Ordering::SeqCst) <= LIMIT);
    assert_eq!(budget.used(), 0);
}

#[test]
fn a_request_larger_than_the_budget_runs_alone() {
    let budget = Arc::new(Budget::new(4));
    let small = budget.acquire_unless(2, None, || false).unwrap();
    let (granted, wait_for_grant) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let oversize = {
        let budget = budget.clone();
        thread::spawn(move || {
            let permit = budget.acquire_unless(100, None, || false).unwrap();
            granted.send((permit.units(), budget.used())).unwrap();
            released.recv().unwrap();
        })
    };
    // It waits for the small holder instead of squeezing in beside it.
    assert!(
        wait_for_grant
            .recv_timeout(Duration::from_millis(150))
            .is_err()
    );
    drop(small);
    // Granted the whole budget, and nothing else is let in beside it.
    assert_eq!(
        wait_for_grant.recv_timeout(Duration::from_secs(5)).unwrap(),
        (4, 4)
    );
    let (second_granted, second_wait) = mpsc::channel();
    let second = {
        let budget = budget.clone();
        thread::spawn(move || {
            let _permit = budget.acquire_unless(1, None, || false).unwrap();
            second_granted.send(()).unwrap();
        })
    };
    assert!(
        second_wait
            .recv_timeout(Duration::from_millis(150))
            .is_err()
    );
    release.send(()).unwrap();
    second_wait.recv_timeout(Duration::from_secs(5)).unwrap();
    oversize.join().unwrap();
    second.join().unwrap();
    assert_eq!(budget.used(), 0);
}

#[test]
fn a_large_request_is_not_starved_by_small_ones() {
    let budget = Arc::new(Budget::new(4));
    let order = Arc::new(Mutex::new(Vec::new()));
    let holder = budget.acquire_unless(3, None, || false).unwrap();
    let large = {
        let (budget, order) = (budget.clone(), order.clone());
        thread::spawn(move || {
            let _permit = budget.acquire_unless(4, None, || false).unwrap();
            order.lock().unwrap().push("large");
        })
    };
    while budget.lock().queue.is_empty() {
        thread::sleep(Duration::from_millis(5));
    }
    // One unit is free, but the large request is ahead in line.
    let small = {
        let (budget, order) = (budget.clone(), order.clone());
        thread::spawn(move || {
            let _permit = budget.acquire_unless(1, None, || false).unwrap();
            order.lock().unwrap().push("small");
        })
    };
    thread::sleep(Duration::from_millis(150));
    assert!(order.lock().unwrap().is_empty());
    drop(holder);
    large.join().unwrap();
    small.join().unwrap();
    assert_eq!(*order.lock().unwrap(), ["large", "small"]);
}

#[test]
fn cancellation_wakes_a_waiter() {
    let budget = Arc::new(Budget::new(1));
    let holder = budget.acquire_unless(1, None, || false).unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    let (ended, wait_for_end) = mpsc::channel();
    let waiter = {
        let (budget, cancelled) = (budget.clone(), cancelled.clone());
        thread::spawn(move || {
            let result = budget
                .acquire_unless(1, None, || cancelled.load(Ordering::SeqCst))
                .map(|_| ());
            ended.send(result).unwrap();
        })
    };
    thread::sleep(Duration::from_millis(100));
    cancelled.store(true, Ordering::SeqCst);
    budget.wake();
    assert_eq!(
        wait_for_end.recv_timeout(Duration::from_secs(5)).unwrap(),
        Err(WaitEnded::Cancelled)
    );
    waiter.join().unwrap();
    // The cancelled request left the line, so the budget is still usable.
    drop(holder);
    assert!(budget.acquire_unless(1, None, || false).is_ok());
}

#[test]
fn a_deadline_ends_a_wait() {
    let budget = Budget::new(1);
    let _holder = budget.acquire_unless(1, None, || false).unwrap();
    let start = Instant::now();
    let ended = budget
        .acquire_unless(1, Some(start + Duration::from_millis(80)), || false)
        .map(|_| ());
    assert_eq!(ended, Err(WaitEnded::TimedOut));
    assert!(start.elapsed() >= Duration::from_millis(80));
    assert_eq!(budget.lock().queue.len(), 0);
}

#[test]
fn a_condition_wait_ends_when_the_condition_clears() {
    let gate = Arc::new(Budget::new(1));
    let blocked = Arc::new(AtomicBool::new(true));
    let waiter = {
        let (gate, blocked) = (gate.clone(), blocked.clone());
        thread::spawn(move || gate.wait_while(None, || blocked.load(Ordering::SeqCst)))
    };
    thread::sleep(Duration::from_millis(100));
    blocked.store(false, Ordering::SeqCst);
    gate.wake();
    assert_eq!(waiter.join().unwrap(), Ok(()));
    assert_eq!(
        gate.wait_while(Some(Instant::now() + Duration::from_millis(50)), || true),
        Err(WaitEnded::TimedOut)
    );
}

#[test]
fn file_workers_and_their_candidates_do_not_deadlock() {
    // The shape of a batch: file workers reserve memory for their image,
    // do a step of their own, then wait on candidate threads that each ask
    // for several units. A file worker holds no work units while it waits.
    finishes_within(Duration::from_secs(60), || {
        let work = Arc::new(Budget::new(4));
        let memory = Arc::new(Budget::new(10));
        let files = (0..8_u64)
            .map(|file| {
                let (work, memory) = (work.clone(), memory.clone());
                thread::spawn(move || {
                    let mut state = 0xdead_beef_u64 ^ (file + 1);
                    let _image = memory.acquire_unless(4, None, || false).unwrap();
                    drop(work.acquire_unless(1, None, || false).unwrap());
                    for _ in 0..5 {
                        let candidates = (0..6)
                            .map(|_| {
                                let work = work.clone();
                                let units = next_random(&mut state) % 7 + 1;
                                thread::spawn(move || {
                                    let _threads =
                                        work.acquire_unless(units, None, || false).unwrap();
                                    thread::sleep(Duration::from_millis(1));
                                })
                            })
                            .collect::<Vec<_>>();
                        for candidate in candidates {
                            candidate.join().unwrap();
                        }
                    }
                    drop(work.acquire_unless(1, None, || false).unwrap());
                })
            })
            .collect::<Vec<_>>();
        for file in files {
            file.join().unwrap();
        }
        assert_eq!(work.used(), 0);
        assert_eq!(memory.used(), 0);
    });
}

#[cfg(windows)]
#[test]
fn process_runner_drains_large_live_output() {
    let output = run_process(
        Path::new("powershell.exe"),
        &[
            OsString::from("-NoProfile"),
            OsString::from("-Command"),
            OsString::from("[Console]::Out.Write('x' * 100000)"),
        ],
        // PowerShell alone can take over ten seconds to start on a loaded CI runner.
        Duration::from_secs(60),
    )
    .unwrap();

    assert_eq!(output.stdout.len(), MAX_PROCESS_OUTPUT_BYTES);
    assert!(output.stdout_truncated);
}

#[cfg(windows)]
#[test]
fn process_runner_starts_tools_in_their_own_folder() {
    let system = PathBuf::from(std::env::var_os("SystemRoot").unwrap());
    let shell = system.join(r"System32\WindowsPowerShell\v1.0\powershell.exe");
    let output = run_process(
        &shell,
        &[
            OsString::from("-NoProfile"),
            OsString::from("-Command"),
            OsString::from("[Console]::Out.Write((Get-Location).Path)"),
        ],
        // PowerShell alone can take over ten seconds to start on a loaded CI runner.
        Duration::from_secs(60),
    )
    .unwrap();

    assert_eq!(
        String::from_utf8_lossy(&output.stdout).to_lowercase(),
        shell.parent().unwrap().to_string_lossy().to_lowercase()
    );
}

#[cfg(windows)]
#[test]
fn process_runner_returns_structured_exit_failure() {
    let error = run_process(
        Path::new("powershell.exe"),
        &[
            OsString::from("-NoProfile"),
            OsString::from("-Command"),
            OsString::from("[Console]::Error.WriteLine('useful failure'); exit 7"),
        ],
        // PowerShell alone can take over ten seconds to start on a loaded CI runner.
        Duration::from_secs(60),
    )
    .unwrap_err();
    let failure = error.downcast_ref::<ProcessFailure>().unwrap();

    assert_eq!(failure.kind, ProcessFailureKind::Exit);
    assert_eq!(failure.exit_code, Some(7));
    assert_eq!(failure.message, "useful failure");
    assert!(!failure.termination_requested);
}

#[cfg(windows)]
#[test]
fn process_runner_times_out_and_requests_tree_termination() {
    let error = run_process(
        Path::new("powershell.exe"),
        &[
            OsString::from("-NoProfile"),
            OsString::from("-Command"),
            OsString::from("Start-Sleep -Seconds 30"),
        ],
        Duration::from_millis(150),
    )
    .unwrap_err();
    let failure = error.downcast_ref::<ProcessFailure>().unwrap();

    assert_eq!(failure.kind, ProcessFailureKind::Timeout);
    assert!(failure.termination_requested);
    assert!(failure.elapsed_ms >= 150);
}

#[cfg(windows)]
#[test]
fn process_runner_terminates_spawned_children() {
    let script = "$child = Start-Process powershell.exe -WindowStyle Hidden -ArgumentList '-NoProfile -Command Start-Sleep -Seconds 300' -PassThru; [Console]::Out.WriteLine($child.Id); Start-Sleep -Seconds 300";
    let error = run_process(
        Path::new("powershell.exe"),
        &[
            OsString::from("-NoProfile"),
            OsString::from("-Command"),
            OsString::from(script),
        ],
        // Long enough for PowerShell to start and print the child id on a
        // slow machine; the script itself sleeps far longer.
        Duration::from_secs(25),
    )
    .unwrap_err();
    let failure = error.downcast_ref::<ProcessFailure>().unwrap();
    let child_id = failure.message.trim().parse::<u32>().unwrap();

    // Each check starts PowerShell, which alone can take over a second
    // on a loaded machine.
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let status = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "if (Get-Process -Id {child_id} -ErrorAction SilentlyContinue) {{ exit 1 }}"
                ),
            ])
            .status()
            .unwrap();
        if status.success() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "spawned child {child_id} survived timeout"
        );
        thread::sleep(Duration::from_millis(50));
    }
}
