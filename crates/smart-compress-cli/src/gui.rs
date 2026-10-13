//! Local control panel: a single page served on the loopback interface that
//! drives the same argument parser and optimizer as the command line.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{Value, json};
use smart_compress_codecs::{Codec, EncoderSettings, ToolInventory, quiet, system_program};
use smart_compress_core::{Optimizer, ProgressObserver, config::GpuConfig, gpu::detect_gpu};

use crate::{Args, build_config, publish_report, run_utility};

const PAGE: &str = include_str!("gui.html");
const IMAGE_EXTENSIONS: [&str; 6] = ["png", "jpg", "jpeg", "webp", "avif", "jxl"];
const INPUT_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "webp", "avif"];
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;
const MAX_UPLOAD_BYTES: usize = 1024 * 1024 * 1024;
const DROP_DIR_PREFIX: &str = "smart-compress-drops-";
/// Hex digits after the prefix: `random_hex(8)`.
const DROP_DIR_ID_LEN: usize = 16;
/// A drop folder without a lock this young may be one being created right now.
const DROP_DIR_GRACE: Duration = Duration::from_secs(60);
/// Locked by the instance that owns a drop folder for as long as it runs.
const DROP_LOCK_NAME: &str = ".in-use";
/// The page itself needs a handful: its own calls and a few thumbnails.
const MAX_CONNECTIONS: usize = 16;
const HEADER_DEADLINE: Duration = Duration::from_secs(10);
const IDLE_TIMEOUT: Duration = Duration::from_secs(10);
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct RunState {
    running: bool,
    started: Option<Instant>,
    elapsed_ms: u64,
    total: usize,
    done: usize,
    files: BTreeMap<String, FileProgress>,
    report: Option<Value>,
    error: Option<String>,
}

#[derive(Default)]
struct FileProgress {
    index: usize,
    total: usize,
    label: String,
}

struct Gui {
    token: String,
    port: u16,
    state: Mutex<RunState>,
    liveness: Mutex<Liveness>,
    connections: AtomicUsize,
    drop_dir: PathBuf,
    /// Taken, and so released, before the drop folder is removed.
    drop_lock: Mutex<Option<File>>,
}

/// What the page last told us, used to stop the server once its window is gone.
struct Liveness {
    last_request: Instant,
    closed_at: Option<Instant>,
}

impl Gui {
    fn state(&self) -> std::sync::MutexGuard<'_, RunState> {
        self.state.lock().expect("GUI state lock poisoned")
    }
}

impl ProgressObserver for Gui {
    fn batch_started(&self, total_files: usize) {
        self.state().total = total_files;
    }

    fn file_started(&self, path: &Path) {
        self.state()
            .files
            .insert(path.display().to_string(), FileProgress::default());
    }

    fn file_finished(&self, path: &Path) {
        let mut state = self.state();
        state.files.remove(&path.display().to_string());
        state.done += 1;
    }

    fn candidates_planned(&self, path: &Path, total_candidates: usize) {
        if let Some(file) = self.state().files.get_mut(&path.display().to_string()) {
            file.total = total_candidates;
        }
    }

    fn candidate_started(
        &self,
        path: &Path,
        _index: usize,
        total: usize,
        codec: Codec,
        settings: &EncoderSettings,
    ) {
        if let Some(file) = self.state().files.get_mut(&path.display().to_string()) {
            file.total = file.total.max(total);
            file.label = format!("{} {}", codec.name(), settings.label());
        }
    }

    fn candidate_finished(&self, path: &Path, _index: usize, _total: usize) {
        // Candidates finish out of order, so count them instead of trusting the index.
        if let Some(file) = self.state().files.get_mut(&path.display().to_string()) {
            file.index += 1;
        }
    }

    fn candidates_settled(&self, path: &Path) {
        if let Some(file) = self.state().files.get_mut(&path.display().to_string()) {
            file.total = file.index;
        }
    }
}

pub fn serve(port: u16, window: bool, open: bool) -> Result<()> {
    // A busy port usually means another panel is open; take the next free one.
    let (listener, port) = (port..=port.saturating_add(20))
        .find_map(|port| {
            TcpListener::bind((Ipv4Addr::LOCALHOST, port))
                .ok()
                .map(|listener| (listener, port))
        })
        .with_context(|| format!("no free port from {port}; pass --gui-port"))?;
    let (drop_dir, drop_lock) = (0..3)
        .find_map(|_| claim_drop_dir().ok())
        .context("creating the folder for dropped files")?;
    remove_stale_drop_dirs(&std::env::temp_dir(), &drop_dir);
    let gui = Arc::new(Gui {
        token: random_hex(16)?,
        port,
        state: Mutex::new(RunState::default()),
        liveness: Mutex::new(Liveness {
            last_request: Instant::now(),
            closed_at: None,
        }),
        connections: AtomicUsize::new(0),
        drop_dir,
        drop_lock: Mutex::new(Some(drop_lock)),
    });
    let url = format!("http://127.0.0.1:{port}/?t={}", gui.token);
    println!("smart-compress control panel: {url}");
    println!("Press Ctrl+C to stop.");
    if window {
        if open {
            open_app_window(&url);
        }
        let gui = Arc::clone(&gui);
        std::thread::spawn(move || exit_when_window_is_gone(&gui));
    } else if open {
        open_in_browser(&url);
    }
    accept_connections(&listener, &gui);
    Ok(())
}

/// One thread per connection, at most `MAX_CONNECTIONS` at a time.
fn accept_connections(listener: &TcpListener, gui: &Arc<Gui>) {
    struct Slot(Arc<Gui>);
    impl Drop for Slot {
        fn drop(&mut self) {
            self.0.connections.fetch_sub(1, Ordering::SeqCst);
        }
    }
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        if gui.connections.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
            gui.connections.fetch_sub(1, Ordering::SeqCst);
            let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
            let _ = respond(
                &mut stream,
                "503 Service Unavailable",
                "text/plain; charset=utf-8",
                b"too many connections",
            );
            // Closing with the request still unread would reset the connection
            // and the client might never see the reply.
            let _ = stream.set_read_timeout(Some(Duration::from_millis(20)));
            let _ = stream.read(&mut [0_u8; 1024]);
            continue;
        }
        let slot = Slot(Arc::clone(gui));
        std::thread::spawn(move || {
            let _ = handle(&slot.0, stream);
        });
    }
}

/// The token keeps other local pages and processes from driving the panel, so
/// it comes from the operating system's random generator.
fn random_hex(bytes: usize) -> Result<String> {
    let mut buffer = vec![0_u8; bytes];
    getrandom::fill(&mut buffer).context("reading the system random generator")?;
    Ok(buffer.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn tokens_match(given: &str, expected: &str) -> bool {
    given.len() == expected.len()
        && given
            .bytes()
            .zip(expected.bytes())
            .fold(0, |difference, (a, b)| difference | (a ^ b))
            == 0
}

fn open_in_browser(url: &str) {
    #[cfg(windows)]
    let _ = quiet(Command::new(system_program(r"System32\cmd.exe")).args(["/C", "start", "", url]))
        .spawn();
    #[cfg(not(windows))]
    let _ = url;
}

/// Edge ships with Windows; its app mode is a plain window without tabs or
/// an address bar. Anything else falls back to a normal browser tab.
fn open_app_window(url: &str) {
    #[cfg(windows)]
    if let Some(edge) = edge_path()
        && Command::new(edge)
            .args([&format!("--app={url}"), "--window-size=1320,900"])
            .spawn()
            .is_ok()
    {
        return;
    }
    open_in_browser(url);
}

/// Edge by its installed path. Asking `cmd` to start `msedge` would also run
/// an `msedge.cmd` or `msedge.exe` sitting in the working directory.
#[cfg(windows)]
fn edge_path() -> Option<PathBuf> {
    ["ProgramFiles(x86)", "ProgramFiles"]
        .into_iter()
        .filter_map(std::env::var_os)
        .map(|dir| PathBuf::from(dir).join(r"Microsoft\Edge\Application\msedge.exe"))
        .find(|path| path.is_absolute() && path.is_file())
}

/// In window mode nobody can press Ctrl+C, so the server ends itself: a few
/// seconds after the page says goodbye (a reload reconnects sooner), or when
/// the page has been silent for minutes because the browser died.
fn exit_when_window_is_gone(gui: &Gui) {
    const CLOSE_GRACE: Duration = Duration::from_secs(4);
    const SILENCE_LIMIT: Duration = Duration::from_secs(300);
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let gone = {
            let liveness = gui.liveness.lock().expect("liveness lock poisoned");
            liveness
                .closed_at
                .is_some_and(|closed| closed.elapsed() >= CLOSE_GRACE)
                || liveness.last_request.elapsed() >= SILENCE_LIMIT
        };
        if !gone {
            continue;
        }
        if gui.state().running {
            // Stop the encoders first; exit on a later pass once they are down.
            smart_compress_codecs::request_cancel();
        } else {
            remove_drop_dir(gui);
            std::process::exit(0);
        }
    }
}

struct Request {
    method: String,
    path: String,
    query: BTreeMap<String, String>,
    token: Option<String>,
    host: Option<String>,
    content_type: Option<String>,
    content_length: usize,
    /// Body bytes that arrived together with the headers.
    received: Vec<u8>,
}

/// Reads up to the end of the headers, within `HEADER_DEADLINE`. The body stays
/// on the socket until the request has been accepted.
fn read_head(stream: &mut TcpStream) -> Result<Request> {
    let deadline = Instant::now() + HEADER_DEADLINE;
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 8192];
    let header_end = loop {
        if let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
        anyhow::ensure!(
            buffer.len() <= MAX_HEADER_BYTES,
            "request headers too large"
        );
        let read = read_before(stream, &mut chunk, deadline)?;
        buffer.extend_from_slice(&chunk[..read]);
    };
    let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let mut lines = head.lines();
    let mut request_line = lines.next().unwrap_or_default().split_whitespace();
    let method = request_line.next().unwrap_or_default().to_string();
    let target = request_line.next().unwrap_or_default();
    let (path, query_text) = target.split_once('?').unwrap_or((target, ""));
    let query = query_text
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .map(|(key, value)| (percent_decode(key), percent_decode(value)))
        .collect::<BTreeMap<_, _>>();
    let mut content_length = 0_usize;
    let mut token = query.get("t").cloned();
    let mut host = None;
    let mut content_type = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => content_length = value.parse().unwrap_or(0),
            "x-token" => token = Some(value.to_string()),
            "host" => host = Some(value.to_string()),
            "content-type" => content_type = Some(value.to_string()),
            _ => {}
        }
    }
    Ok(Request {
        method,
        path: path.to_string(),
        query,
        token,
        host,
        content_type,
        content_length,
        received: buffer[header_end..].to_vec(),
    })
}

/// One read that neither outlasts the deadline nor waits longer than
/// `IDLE_TIMEOUT` for the peer.
fn read_before(stream: &mut TcpStream, chunk: &mut [u8], deadline: Instant) -> Result<usize> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    anyhow::ensure!(!remaining.is_zero(), "request took too long");
    stream.set_read_timeout(Some(remaining.min(IDLE_TIMEOUT)))?;
    let read = stream.read(chunk)?;
    anyhow::ensure!(read > 0, "connection closed before the request completed");
    Ok(read)
}

fn read_body(stream: &mut TcpStream, request: &mut Request) -> Result<Vec<u8>> {
    // Thirty seconds plus one per MiB: a slower sender is not a browser.
    let deadline = Instant::now() + Duration::from_secs(30 + (request.content_length >> 20) as u64);
    let mut body = std::mem::take(&mut request.received);
    let mut chunk = [0_u8; 64 * 1024];
    while body.len() < request.content_length {
        let read = read_before(stream, &mut chunk, deadline)?;
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(request.content_length);
    Ok(body)
}

fn host_allowed(host: Option<&str>, port: u16) -> bool {
    host.is_some_and(|host| {
        host.eq_ignore_ascii_case(&format!("127.0.0.1:{port}"))
            || host.eq_ignore_ascii_case(&format!("localhost:{port}"))
    })
}

fn is_json(content_type: Option<&str>) -> bool {
    content_type
        .and_then(|value| value.split(';').next())
        .is_some_and(|media| media.trim().eq_ignore_ascii_case("application/json"))
}

/// UNC and device paths (`\\server\share`, `\\?\UNC\...`, `\\.\...`) make
/// Windows contact the host they name. Drive letters, mapped ones too, are
/// unaffected.
fn is_network_or_device_path(path: &str) -> bool {
    let mut characters = path.chars();
    matches!(
        (characters.next(), characters.next()),
        (Some('\\' | '/'), Some('\\' | '/'))
    )
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let hex = bytes
            .get(index + 1..index + 3)
            .and_then(|pair| std::str::from_utf8(pair).ok())
            .and_then(|pair| u8::from_str_radix(pair, 16).ok());
        match (bytes[index], hex) {
            (b'%', Some(byte)) => {
                decoded.push(byte);
                index += 3;
            }
            (b'+', _) => {
                decoded.push(b' ');
                index += 1;
            }
            (byte, _) => {
                decoded.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&decoded).to_string()
}

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) -> Result<()> {
    respond_with(stream, status, content_type, "", body)
}

/// `extra` is whole header lines, each ending in CRLF. The token travels in the
/// page's address, so no response may pass it on as a referrer.
fn respond_with(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    extra: &str,
    body: &[u8],
) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n{extra}Connection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    Ok(())
}

/// The page is one file with an inline script and style, each carrying this
/// response's nonce; it talks only to its own origin.
fn respond_page(stream: &mut TcpStream) -> Result<()> {
    let nonce = random_hex(16)?;
    let policy = format!(
        "Content-Security-Policy: default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'\r\n"
    );
    respond_with(
        stream,
        "200 OK",
        "text/html; charset=utf-8",
        &policy,
        PAGE.replace("{{nonce}}", &nonce).as_bytes(),
    )
}

fn respond_json(stream: &mut TcpStream, status: &str, value: &Value) -> Result<()> {
    respond(
        stream,
        status,
        "application/json; charset=utf-8",
        value.to_string().as_bytes(),
    )
}

fn handle(gui: &Arc<Gui>, mut stream: TcpStream) -> Result<()> {
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    let mut request = read_head(&mut stream)?;
    // All of this is decided before the body is read, so a refused request
    // costs the server nothing.
    if !host_allowed(request.host.as_deref(), gui.port) {
        return respond(
            &mut stream,
            "403 Forbidden",
            "text/plain; charset=utf-8",
            b"Unexpected Host header.",
        );
    }
    if !tokens_match(request.token.as_deref().unwrap_or_default(), &gui.token) {
        return respond(
            &mut stream,
            "403 Forbidden",
            "text/plain; charset=utf-8",
            b"Open the link printed by `smart-compress --gui`; it carries this session's access token.",
        );
    }
    let is_upload = (request.method.as_str(), request.path.as_str()) == ("POST", "/api/upload");
    let body_limit = if is_upload {
        MAX_UPLOAD_BYTES
    } else {
        MAX_BODY_BYTES
    };
    if request.content_length > body_limit {
        return respond_json(
            &mut stream,
            "413 Payload Too Large",
            &json!({ "error": "request body too large" }),
        );
    }
    if request.method == "POST" && !is_upload && !is_json(request.content_type.as_deref()) {
        return respond_json(
            &mut stream,
            "415 Unsupported Media Type",
            &json!({ "error": "requests must be application/json" }),
        );
    }
    let body_bytes = if request.method == "POST" {
        read_body(&mut stream, &mut request)?
    } else {
        Vec::new()
    };
    if is_upload {
        let saved = save_upload(gui, request.query.get("name"), &body_bytes);
        return match saved {
            Ok(value) => respond_json(&mut stream, "200 OK", &value),
            Err(error) => respond_json(
                &mut stream,
                "400 Bad Request",
                &json!({ "error": format!("{error:#}") }),
            ),
        };
    }
    {
        let mut liveness = gui.liveness.lock().expect("liveness lock poisoned");
        liveness.last_request = Instant::now();
        liveness.closed_at = (request.path == "/api/bye").then_some(liveness.last_request);
    }
    let body: Value = if body_bytes.is_empty() {
        Value::Null
    } else {
        match serde_json::from_slice(&body_bytes) {
            Ok(value) => value,
            Err(error) => {
                return respond_json(
                    &mut stream,
                    "400 Bad Request",
                    &json!({ "error": format!("invalid JSON body: {error}") }),
                );
            }
        }
    };
    let result = match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") => return respond_page(&mut stream),
        ("GET", "/api/file") => return serve_image(&mut stream, request.query.get("path")),
        ("GET", "/api/info") => Ok(info()),
        ("GET", "/api/status") => Ok(status(gui)),
        ("POST", "/api/pick") => pick(body["kind"].as_str().unwrap_or("files")),
        ("POST", "/api/stat") => Ok(stat(&string_list(&body["paths"]))),
        ("POST", "/api/run") => start_run(gui, string_list(&body["argv"])),
        ("POST", "/api/bye") => Ok(json!({ "ok": true })),
        ("POST", "/api/cancel") => {
            smart_compress_codecs::request_cancel();
            Ok(json!({ "ok": true }))
        }
        ("POST", "/api/open") => reveal(body["path"].as_str().unwrap_or_default()),
        _ => {
            return respond_json(
                &mut stream,
                "404 Not Found",
                &json!({ "error": "unknown endpoint" }),
            );
        }
    };
    match result {
        Ok(value) => respond_json(&mut stream, "200 OK", &value),
        Err(error) => respond_json(
            &mut stream,
            "400 Bad Request",
            &json!({ "error": format!("{error:#}") }),
        ),
    }
}

fn string_list(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn has_extension(path: &Path, allowed: &[&str]) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| allowed.contains(&extension.to_ascii_lowercase().as_str()))
}

/// A folder of this instance's own, named at random, with a lock on a file in
/// it for as long as the process lives (the system drops it if the process dies).
fn claim_drop_dir() -> Result<(PathBuf, File)> {
    let dir = std::env::temp_dir().join(format!("{DROP_DIR_PREFIX}{}", random_hex(8)?));
    #[cfg_attr(not(unix), allow(unused_mut))]
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(&dir)?;
    let lock = File::options()
        .write(true)
        .create_new(true)
        .open(dir.join(DROP_LOCK_NAME))?;
    lock.try_lock()?;
    Ok((dir, lock))
}

/// Whether the entry is a link of any kind (symlink, junction, other reparse point).
fn is_link(metadata: &std::fs::Metadata) -> bool {
    #[cfg(windows)]
    let reparse_point = {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    };
    #[cfg(not(windows))]
    let reparse_point = false;
    metadata.file_type().is_symlink() || reparse_point
}

/// Dropped files from earlier sessions are copies; the originals are untouched.
/// A folder whose lock is held belongs to an instance that is still running.
/// Only real folders with exactly the name this program generates are touched,
/// and the lock is opened, never created: the temporary folder may be shared,
/// and anything else in it, links included, is not ours. `own` is this
/// instance's folder, whose owner other folders must have (checked on Unix).
fn remove_stale_drop_dirs(temp: &Path, own: &Path) {
    let Ok(entries) = std::fs::read_dir(temp) else {
        return;
    };
    #[cfg(unix)]
    let me = {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(own).map(|metadata| metadata.uid()).ok()
    };
    #[cfg(not(unix))]
    let _ = own;
    for entry in entries.flatten() {
        let path = entry.path();
        let named_like_ours = entry
            .file_name()
            .to_string_lossy()
            .strip_prefix(DROP_DIR_PREFIX)
            .is_some_and(|id| {
                id.len() == DROP_DIR_ID_LEN
                    && id
                        .bytes()
                        .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
            });
        let Some(folder) = std::fs::symlink_metadata(&path)
            .ok()
            .filter(|metadata| named_like_ours && metadata.is_dir() && !is_link(metadata))
        else {
            continue;
        };
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if Some(folder.uid()) != me {
                continue;
            }
        }
        let lock_path = path.join(DROP_LOCK_NAME);
        match std::fs::symlink_metadata(&lock_path) {
            Ok(lock) if lock.is_file() && !is_link(&lock) => {
                // No `create` and no `truncate`: this only asks for the lock.
                let Ok(lock) = File::options().write(true).open(&lock_path) else {
                    continue;
                };
                if lock.try_lock().is_err() {
                    continue;
                }
            }
            // No lock at all: a start that crashed, or an older version. Not a
            // folder that is just being made.
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound
                    && folder
                        .modified()
                        .ok()
                        .and_then(|modified| modified.elapsed().ok())
                        .is_some_and(|age| age > DROP_DIR_GRACE) => {}
            _ => continue,
        }
        // Does not follow links inside the folder: std unlinks them (Rust 1.58.1 on).
        let _ = std::fs::remove_dir_all(&path);
    }
}

fn remove_drop_dir(gui: &Gui) {
    drop(gui.drop_lock.lock().expect("drop lock poisoned").take());
    let _ = std::fs::remove_dir_all(&gui.drop_dir);
}

/// Browsers hide the path of a dropped file, so the bytes are copied to a
/// session folder and that copy becomes the input.
fn save_upload(gui: &Gui, name: Option<&String>, bytes: &[u8]) -> Result<Value> {
    let name = name
        .map(Path::new)
        .and_then(Path::file_name)
        .context("dropped file has no name")?;
    anyhow::ensure!(
        has_extension(Path::new(name), &INPUT_EXTENSIONS),
        "{} is not a supported image (PNG, JPEG, WebP, AVIF)",
        name.to_string_lossy()
    );
    let dir = &gui.drop_dir;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let stem = Path::new(name)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let extension = Path::new(name)
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let path = (0..1000)
        .map(|attempt| match attempt {
            0 => dir.join(name),
            _ => dir.join(format!("{stem}-{attempt}.{extension}")),
        })
        .find(|path| !path.exists())
        .context("too many dropped files share this name")?;
    std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(json!({ "path": path }))
}

fn drop_output_dir() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .map(|home| PathBuf::from(home).join("Downloads"))
        .filter(|downloads| downloads.is_dir())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
        .join("smart-compress")
}

fn info() -> Value {
    static INFO: OnceLock<Value> = OnceLock::new();
    INFO.get_or_init(|| {
        json!({
            "version": env!("CARGO_PKG_VERSION"),
            "cwd": std::env::current_dir().unwrap_or_default(),
            "workers": std::thread::available_parallelism().map_or(1, usize::from),
            "native_dialogs": cfg!(windows),
            "drop_output": drop_output_dir(),
            "tools": ToolInventory::detect(),
            "gpu": detect_gpu(&GpuConfig::default()),
        })
    })
    .clone()
}

fn status(gui: &Gui) -> Value {
    let state = gui.state();
    json!({
        "running": state.running,
        "elapsed_ms": state
            .started
            .map_or(state.elapsed_ms, |started| started.elapsed().as_millis() as u64),
        "total": state.total,
        "done": state.done,
        "files": state
            .files
            .iter()
            .map(|(path, file)| json!({
                "path": path,
                "index": file.index,
                "total": file.total,
                "label": file.label,
            }))
            .collect::<Vec<_>>(),
        "report": if state.running { None } else { state.report.as_ref() },
        "error": state.error,
    })
}

fn start_run(gui: &Arc<Gui>, argv: Vec<String>) -> Result<Value> {
    let args = Args::try_parse_from(std::iter::once("smart-compress".to_string()).chain(argv))
        .map_err(|error| anyhow::anyhow!(error.render().to_string()))?;
    anyhow::ensure!(!args.gui, "--gui cannot be started from the control panel");
    if let Some(message) = run_utility(&args)? {
        return Ok(json!({ "message": message }));
    }
    anyhow::ensure!(!args.input.is_empty(), "add at least one image or folder");
    let config = build_config(args)?;
    {
        let mut state = gui.state();
        anyhow::ensure!(!state.running, "a run is already in progress");
        *state = RunState {
            running: true,
            started: Some(Instant::now()),
            ..RunState::default()
        };
    }
    smart_compress_codecs::clear_cancel();
    let gui = Arc::clone(gui);
    std::thread::spawn(move || {
        let report_format = config.report;
        let report_out = config.report_out.clone();
        let progress: Arc<dyn ProgressObserver> = gui.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let summary = Optimizer::with_progress(config, progress).run()?;
            if let Some(path) = &report_out {
                publish_report(&summary.report, report_format, Some(path))?;
            }
            serde_json::to_value(&summary.report).context("serializing report")
        }));
        let mut state = gui.state();
        state.running = false;
        state.elapsed_ms = state
            .started
            .take()
            .map_or(0, |start| start.elapsed().as_millis() as u64);
        state.files.clear();
        match outcome {
            Ok(Ok(report)) => state.report = Some(report),
            Ok(Err(error)) => state.error = Some(format!("{error:#}")),
            Err(_) => state.error = Some("the optimizer stopped unexpectedly".to_string()),
        }
    });
    Ok(json!({ "ok": true }))
}

fn stat(paths: &[String]) -> Value {
    fn count_images(dir: &Path, depth: usize) -> u64 {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return 0;
        };
        entries
            .flatten()
            .map(|entry| {
                let path = entry.path();
                if path.is_dir() && depth < 32 {
                    count_images(&path, depth + 1)
                } else {
                    u64::from(has_extension(&path, &INPUT_EXTENSIONS))
                }
            })
            .sum()
    }
    Value::Array(
        paths
            .iter()
            .map(|text| {
                let path = Path::new(text);
                if is_network_or_device_path(text) {
                    return json!({ "path": text, "kind": "remote" });
                }
                match std::fs::metadata(path) {
                    Ok(metadata) if metadata.is_dir() => json!({
                        "path": text, "kind": "dir", "count": count_images(path, 0),
                    }),
                    Ok(metadata) => json!({
                        "path": text,
                        "kind": if has_extension(path, &INPUT_EXTENSIONS) { "file" } else { "unsupported" },
                        "size": metadata.len(),
                    }),
                    Err(_) => json!({ "path": text, "kind": "missing" }),
                }
            })
            .collect(),
    )
}

fn serve_image(stream: &mut TcpStream, path: Option<&String>) -> Result<()> {
    let path = PathBuf::from(path.map(String::as_str).unwrap_or_default());
    // Previews only ever need image files; refuse to act as a general file server.
    if !has_extension(&path, &IMAGE_EXTENSIONS) {
        return respond(stream, "403 Forbidden", "text/plain", b"not an image path");
    }
    if is_network_or_device_path(&path.to_string_lossy()) {
        return respond(
            stream,
            "403 Forbidden",
            "text/plain",
            b"network paths are not served",
        );
    }
    let content_type = match path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "png" => "image/png",
        "webp" => "image/webp",
        "avif" => "image/avif",
        "jxl" => "image/jxl",
        _ => "image/jpeg",
    };
    match std::fs::read(&path) {
        Ok(bytes) => respond(stream, "200 OK", content_type, &bytes),
        Err(_) => respond(stream, "404 Not Found", "text/plain", b"file not found"),
    }
}

// Native dialogs go through Windows PowerShell so no GUI toolkit is linked;
// other platforms type or paste paths. A dialog crate would replace this if the
// program ever ships beyond Windows.
fn pick(kind: &str) -> Result<Value> {
    anyhow::ensure!(
        cfg!(windows),
        "file dialogs are only available on Windows; paste paths instead"
    );
    let dialog = match kind {
        "files" => {
            "$d=New-Object Windows.Forms.OpenFileDialog -Property @{Multiselect=$true;Title='Add images';Filter='Images|*.png;*.jpg;*.jpeg;*.webp;*.avif|All files|*.*'};if($d.ShowDialog($o) -eq 'OK'){$d.FileNames}"
        }
        "folder" => {
            "$d=New-Object Windows.Forms.FolderBrowserDialog -Property @{Description='Choose a folder'};if($d.ShowDialog($o) -eq 'OK'){$d.SelectedPath}"
        }
        "save" => {
            "$d=New-Object Windows.Forms.SaveFileDialog -Property @{Title='Save report as';Filter='All files|*.*'};if($d.ShowDialog($o) -eq 'OK'){$d.FileName}"
        }
        "open" => {
            "$d=New-Object Windows.Forms.OpenFileDialog -Property @{Title='Choose a file';Filter='All files|*.*'};if($d.ShowDialog($o) -eq 'OK'){$d.FileName}"
        }
        other => anyhow::bail!("unknown dialog kind '{other}'"),
    };
    let script = format!(
        "Add-Type -AssemblyName System.Windows.Forms;[Console]::OutputEncoding=[Text.Encoding]::UTF8;$o=New-Object Windows.Forms.Form -Property @{{TopMost=$true}};{dialog}"
    );
    let output = quiet(
        Command::new(system_program(
            r"System32\WindowsPowerShell\v1.0\powershell.exe",
        ))
        .args(["-NoProfile", "-STA", "-Command", &script]),
    )
    .output()
    .context("opening the file dialog")?;
    let paths = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    Ok(json!({ "paths": paths }))
}

fn reveal(path: &str) -> Result<Value> {
    anyhow::ensure!(
        cfg!(windows),
        "opening folders is only available on Windows"
    );
    anyhow::ensure!(
        !is_network_or_device_path(path),
        "network and device paths are not opened"
    );
    let path = Path::new(path);
    anyhow::ensure!(path.exists(), "{} does not exist", path.display());
    let folder = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(path)
    };
    Command::new(system_program("explorer.exe"))
        .arg(folder)
        .spawn()
        .context("opening File Explorer")?;
    Ok(json!({ "ok": true }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_decoding_handles_paths() {
        assert_eq!(
            percent_decode("C%3A%5Cmy+photos%5Ca%20b.jpg"),
            "C:\\my photos\\a b.jpg"
        );
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%CE%B1.png"), "α.png");
    }

    const TEST_TOKEN: &str = "0123456789abcdef0123456789abcdef";

    /// A server on a free port; the connections are never closed by the tests.
    fn test_server() -> (u16, Arc<Gui>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let gui = Arc::new(Gui {
            token: TEST_TOKEN.to_string(),
            port,
            state: Mutex::new(RunState::default()),
            liveness: Mutex::new(Liveness {
                last_request: Instant::now(),
                closed_at: None,
            }),
            connections: AtomicUsize::new(0),
            drop_dir: std::env::temp_dir().join("smart-compress-test-unused"),
            drop_lock: Mutex::new(None),
        });
        let served = Arc::clone(&gui);
        std::thread::spawn(move || accept_connections(&listener, &served));
        (port, gui)
    }

    /// Sends `head` as it is, without any body, and returns the reply.
    fn exchange(port: u16, head: &str) -> String {
        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream.write_all(head.as_bytes()).unwrap();
        let mut reply = String::new();
        let _ = stream.read_to_string(&mut reply);
        reply
    }

    #[test]
    fn refused_requests_are_answered_without_reading_the_body() {
        let (port, _) = test_server();
        let post = |extra: &str| {
            format!(
                "POST /api/cancel HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: 1073741824\r\n{extra}\r\n"
            )
        };

        // No token, and a body that was announced but is never sent.
        assert!(exchange(port, &post("")).starts_with("HTTP/1.1 403"));
        // A name that points at the loopback address but is not ours.
        let rebound = post(&format!("X-Token: {TEST_TOKEN}\r\n"))
            .replace(&format!("Host: 127.0.0.1:{port}"), "Host: attacker.example");
        assert!(exchange(port, &rebound).starts_with("HTTP/1.1 403"));
        // Too large for this endpoint.
        let token = format!("X-Token: {TEST_TOKEN}\r\nContent-Type: application/json\r\n");
        assert!(exchange(port, &post(&token)).starts_with("HTTP/1.1 413"));
        // Not JSON.
        let text = format!("X-Token: {TEST_TOKEN}\r\nContent-Type: text/plain\r\n");
        let text = post(&text).replace("1073741824", "2");
        assert!(exchange(port, &text).starts_with("HTTP/1.1 415"));
    }

    #[test]
    fn the_page_and_api_carry_the_security_headers() {
        let (port, _) = test_server();
        let get = |path: &str| {
            exchange(
                port,
                &format!("GET {path}?t={TEST_TOKEN} HTTP/1.1\r\nHost: localhost:{port}\r\n\r\n"),
            )
        };

        let page = get("/");
        assert!(page.starts_with("HTTP/1.1 200"));
        assert!(page.contains("Referrer-Policy: no-referrer"));
        assert!(page.contains("X-Content-Type-Options: nosniff"));
        let nonce = page
            .split("script-src 'nonce-")
            .nth(1)
            .and_then(|rest| rest.split('\'').next())
            .unwrap();
        assert_eq!(nonce.len(), 32);
        assert!(page.contains(&format!("<script nonce=\"{nonce}\">")));
        assert!(page.contains(&format!("<style nonce=\"{nonce}\">")));
        assert!(!page.contains("{{nonce}}"));

        let status = get("/api/status");
        assert!(status.starts_with("HTTP/1.1 200"));
        assert!(status.contains("Referrer-Policy: no-referrer"));
    }

    #[test]
    fn idle_connections_cannot_exhaust_the_server() {
        let (port, gui) = test_server();
        let idle = (0..MAX_CONNECTIONS)
            .map(|_| TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap())
            .collect::<Vec<_>>();
        // Wait until the accept loop has taken all of them.
        let deadline = Instant::now() + Duration::from_secs(10);
        while gui.connections.load(Ordering::SeqCst) < MAX_CONNECTIONS {
            assert!(Instant::now() < deadline, "connections were not accepted");
            std::thread::sleep(Duration::from_millis(10));
        }
        let reply = exchange(port, "GET / HTTP/1.1\r\n\r\n");
        assert!(reply.starts_with("HTTP/1.1 503"), "{reply}");
        drop(idle);
    }

    #[test]
    fn tokens_hosts_and_content_types_are_checked_exactly() {
        assert!(tokens_match(TEST_TOKEN, TEST_TOKEN));
        assert!(!tokens_match(TEST_TOKEN, &TEST_TOKEN.replace('f', "e")));
        assert!(!tokens_match("", TEST_TOKEN));
        assert!(!tokens_match(&TEST_TOKEN[1..], TEST_TOKEN));

        assert!(host_allowed(Some("127.0.0.1:8000"), 8000));
        assert!(host_allowed(Some("LocalHost:8000"), 8000));
        assert!(!host_allowed(Some("127.0.0.1:8001"), 8000));
        assert!(!host_allowed(Some("127.0.0.1"), 8000));
        assert!(!host_allowed(Some("evil.example:8000"), 8000));
        assert!(!host_allowed(None, 8000));

        assert!(is_json(Some("application/json")));
        assert!(is_json(Some("Application/JSON; charset=utf-8")));
        assert!(!is_json(Some("text/plain")));
        assert!(!is_json(Some("application/jsonp")));
        assert!(!is_json(None));
    }

    #[test]
    fn session_tokens_are_random_and_full_length() {
        let (first, second) = (random_hex(16).unwrap(), random_hex(16).unwrap());
        assert_eq!(first.len(), 32);
        assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(first, second);
    }

    #[test]
    fn network_and_device_paths_are_refused() {
        for path in [
            r"\\server\share\a.png",
            r"\\?\UNC\server\share\a.png",
            r"\\.\pipe\x",
            "//server/share/a.png",
            r"/\server\share",
        ] {
            assert!(is_network_or_device_path(path), "{path}");
        }
        for path in [
            r"C:\photos\a.png",
            r"Z:\a.png",
            "a.png",
            r"\a.png",
            "",
            "/a.png",
        ] {
            assert!(!is_network_or_device_path(path), "{path}");
        }
    }

    #[test]
    fn a_drop_folder_in_use_survives_the_sweep() {
        let (dir, lock) = claim_drop_dir().unwrap();
        std::fs::write(dir.join("a.png"), b"x").unwrap();
        let temp = std::env::temp_dir();
        remove_stale_drop_dirs(&temp, &dir);
        assert!(dir.join("a.png").exists());

        // Once the owner is gone, as after a crash, the folder is stale.
        drop(lock);
        remove_stale_drop_dirs(&temp, &dir);
        assert!(!dir.exists());
    }

    /// A link to a folder; false when the system does not let this user make one.
    fn make_dir_link(link: &Path, target: &Path) -> bool {
        #[cfg(windows)]
        return Command::new(system_program(r"System32\cmd.exe"))
            .args(["/C", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .output()
            .is_ok_and(|output| output.status.success());
        #[cfg(unix)]
        return std::os::unix::fs::symlink(target, link).is_ok();
    }

    #[test]
    fn the_sweep_ignores_links_in_the_temporary_folder() {
        let temp = std::env::temp_dir().join(format!("sc-sweep-test-{}", random_hex(8).unwrap()));
        let victim = temp.join("victim");
        std::fs::create_dir_all(&victim).unwrap();
        std::fs::write(victim.join("keep.txt"), b"precious").unwrap();
        let own = temp.join("own");
        std::fs::create_dir(&own).unwrap();
        let name = |tag: char| {
            format!(
                "{DROP_DIR_PREFIX}{}",
                tag.to_string().repeat(DROP_DIR_ID_LEN)
            )
        };

        // A drop folder that is itself a link to a folder with content.
        let linked = temp.join(name('a'));
        let linked_made = make_dir_link(&linked, &victim);
        // A real one whose lock entry is a link to a file.
        let planted = temp.join(name('b'));
        std::fs::create_dir(&planted).unwrap();
        let target = victim.join("keep.txt");
        #[cfg(windows)]
        let file_link_made =
            std::os::windows::fs::symlink_file(&target, planted.join(DROP_LOCK_NAME)).is_ok();
        #[cfg(unix)]
        let file_link_made =
            std::os::unix::fs::symlink(&target, planted.join(DROP_LOCK_NAME)).is_ok();
        // A folder that only looks like one: wrong name.
        let lookalike = temp.join(format!("{DROP_DIR_PREFIX}notes"));
        std::fs::create_dir(&lookalike).unwrap();

        remove_stale_drop_dirs(&temp, &own);

        assert_eq!(std::fs::read(&target).unwrap(), b"precious");
        assert!(planted.exists() && lookalike.exists());
        if linked_made {
            assert!(linked.exists());
            std::fs::remove_dir(&linked).unwrap();
        } else {
            println!("skipped the linked-folder case: links cannot be made here");
        }
        if !file_link_made {
            println!("skipped the linked-lock case: file symlinks cannot be made here");
        }
        assert!(target.exists());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[cfg(windows)]
    #[test]
    fn network_paths_are_not_opened_in_explorer() {
        let error = reveal(r"\\127.0.0.1\share\a.png").unwrap_err();
        assert!(error.to_string().contains("network"), "{error}");
    }

    #[cfg(windows)]
    #[test]
    fn edge_is_started_by_absolute_path() {
        assert!(edge_path().is_none_or(|path| path.is_absolute() && path.is_file()));
    }

    #[test]
    fn only_image_paths_are_served() {
        assert!(has_extension(Path::new("a/b.JPG"), &IMAGE_EXTENSIONS));
        assert!(!has_extension(Path::new("a/.env"), &IMAGE_EXTENSIONS));
        assert!(!has_extension(Path::new("a/id_rsa"), &IMAGE_EXTENSIONS));
    }
}
