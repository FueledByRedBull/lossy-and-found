// Runs a command inside a Windows job object and reports what the whole
// process tree cost: wall time, user+kernel CPU seconds (finished children
// included), average and peak cores in use, and peak working set / commit
// summed over the tree. No dependencies:
//
//   rustc -O --edition 2024 tools/measure-process-tree.rs -o target/measure/measure.exe
//   measure.exe [--stdout F] [--stderr F] [--summary F] [--interval-ms N] -- cmd args...
//
// Pin it like any other command, for example
//   cmd //C start //B //WAIT //AFFINITY 7F target/measure/measure.exe ...
// because children inherit the affinity mask. The summary is one JSON object.
use std::{
    ffi::c_void,
    fs::File,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

type Handle = *mut c_void;

#[repr(C)]
#[derive(Default)]
struct Accounting {
    user: i64,
    kernel: i64,
    this_period_user: i64,
    this_period_kernel: i64,
    page_faults: u32,
    total_processes: u32,
    active_processes: u32,
    terminated_processes: u32,
}

#[repr(C)]
struct PidList {
    assigned: u32,
    listed: u32,
    ids: [usize; 512],
}

#[repr(C)]
#[derive(Default)]
struct MemoryCounters {
    size: u32,
    page_faults: u32,
    peak_working_set: usize,
    working_set: usize,
    rest: [usize; 4],
    pagefile: usize,
    peak_pagefile: usize,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateJobObjectW(attributes: *mut c_void, name: *const u16) -> Handle;
    fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
    fn GetCurrentProcess() -> Handle;
    fn QueryInformationJobObject(
        job: Handle,
        class: i32,
        info: *mut c_void,
        length: u32,
        returned: *mut u32,
    ) -> i32;
    fn OpenProcess(access: u32, inherit: i32, pid: u32) -> Handle;
    fn CloseHandle(handle: Handle) -> i32;
    fn K32GetProcessMemoryInfo(process: Handle, counters: *mut MemoryCounters, size: u32) -> i32;
}

fn cpu_seconds(job: Handle) -> (f64, f64, u32) {
    let mut info = Accounting::default();
    let mut returned = 0_u32;
    unsafe {
        QueryInformationJobObject(
            job,
            1,
            (&mut info as *mut Accounting).cast(),
            size_of::<Accounting>() as u32,
            &mut returned,
        );
    }
    (
        info.user as f64 / 1e7,
        info.kernel as f64 / 1e7,
        info.total_processes,
    )
}

/// Working set and commit summed over the processes alive in the job.
fn memory_bytes(job: Handle) -> (u64, u64, u32) {
    let mut list = PidList {
        assigned: 0,
        listed: 0,
        ids: [0; 512],
    };
    let mut returned = 0_u32;
    unsafe {
        QueryInformationJobObject(
            job,
            3,
            (&mut list as *mut PidList).cast(),
            size_of::<PidList>() as u32,
            &mut returned,
        );
    }
    let (mut working_set, mut commit) = (0_u64, 0_u64);
    for &pid in &list.ids[..(list.listed as usize).min(512)] {
        let process = unsafe { OpenProcess(0x1000, 0, pid as u32) };
        if process.is_null() {
            continue;
        }
        let mut counters = MemoryCounters {
            size: size_of::<MemoryCounters>() as u32,
            ..Default::default()
        };
        if unsafe { K32GetProcessMemoryInfo(process, &mut counters, counters.size) } != 0 {
            working_set += counters.working_set as u64;
            commit += counters.pagefile as u64;
        }
        unsafe { CloseHandle(process) };
    }
    (working_set, commit, list.listed)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut stdout, mut stderr, mut summary) = (None, None, None);
    let mut interval = Duration::from_millis(50);
    let mut command = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--stdout" => stdout = args.next(),
            "--stderr" => stderr = args.next(),
            "--summary" => summary = args.next(),
            "--interval-ms" => {
                interval = Duration::from_millis(args.next().unwrap().parse().unwrap())
            }
            "--" => {
                command.extend(args.by_ref());
            }
            other => panic!("unknown option {other}"),
        }
    }
    let job = unsafe { CreateJobObjectW(std::ptr::null_mut(), std::ptr::null()) };
    assert!(
        unsafe { AssignProcessToJobObject(job, GetCurrentProcess()) } != 0,
        "could not join the job object"
    );
    let open = |path: &Option<String>| match path {
        Some(path) => Stdio::from(File::create(path).unwrap()),
        None => Stdio::inherit(),
    };
    let start = Instant::now();
    let mut child = Command::new(&command[0])
        .args(&command[1..])
        .stdout(open(&stdout))
        .stderr(open(&stderr))
        .spawn()
        .expect("starting the command");

    let (mut peak_ws, mut peak_commit, mut peak_processes) = (0_u64, 0_u64, 0_u32);
    let mut history = vec![(0.0_f64, 0.0_f64)];
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        std::thread::sleep(interval);
        let (ws, commit, processes) = memory_bytes(job);
        peak_ws = peak_ws.max(ws);
        peak_commit = peak_commit.max(commit);
        peak_processes = peak_processes.max(processes);
        let (user, kernel, _) = cpu_seconds(job);
        history.push((start.elapsed().as_secs_f64(), user + kernel));
    };
    let wall = start.elapsed().as_secs_f64();
    let (user, kernel, total_processes) = cpu_seconds(job);
    let cpu = user + kernel;
    history.push((wall, cpu));
    // Peak cores: the busiest window of at least one second.
    let mut peak_cores = 0.0_f64;
    let mut low = 0;
    for high in 0..history.len() {
        while low + 1 < high && history[high].0 - history[low + 1].0 >= 1.0 {
            low += 1;
        }
        let span = history[high].0 - history[low].0;
        if span >= 1.0 {
            peak_cores = peak_cores.max((history[high].1 - history[low].1) / span);
        }
    }
    let text = format!(
        "{{\"exit\":{},\"wall_s\":{wall:.3},\"cpu_user_s\":{user:.3},\"cpu_kernel_s\":{kernel:.3},\"cpu_s\":{cpu:.3},\"avg_cores\":{:.3},\"peak_cores_1s\":{peak_cores:.3},\"peak_working_set_mb\":{:.0},\"peak_commit_mb\":{:.0},\"peak_processes\":{peak_processes},\"total_processes\":{total_processes}}}",
        status.code().unwrap_or(-1),
        cpu / wall,
        peak_ws as f64 / 1048576.0,
        peak_commit as f64 / 1048576.0,
    );
    match summary {
        Some(path) => std::fs::write(path, text).unwrap(),
        None => eprintln!("{text}"),
    }
    std::process::exit(status.code().unwrap_or(1));
}
