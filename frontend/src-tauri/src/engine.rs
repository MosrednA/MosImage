//! Starts the local Python engine with the app, restarts it after a hard cancel
//! or crash, and stops it cleanly on exit, like backend/supervisor.ps1 does.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const HOST: &str = "127.0.0.1";
/// MOSIMAGE_PORT at build time, else 8000; the UI build reads the same variable.
const PORT: u16 = match option_env!("MOSIMAGE_PORT") {
    Some(port) => parse_port(port),
    None => 8000,
};
/// Cancelling a job and unloading the models takes a few seconds; after this the engine is ended.
const STOP_GRACE: Duration = Duration::from_secs(20);
const RESTART_DELAY: Duration = Duration::from_secs(1);
/// An engine that keeps failing this soon after starting is broken, not cancelled.
const QUICK_FAILURE: Duration = Duration::from_secs(15);
const MAX_QUICK_FAILURES: u32 = 3;

pub struct Engine {
    state: Option<Arc<Mutex<Supervised>>>,
}

/// An engine this app started, and may therefore restart and stop.
struct Supervised {
    root: PathBuf,
    token: String,
    child: Option<Child>,
    started_at: Instant,
    quick_failures: u32,
    stopping: bool,
    #[cfg(windows)]
    job: Option<job::Job>,
}

impl Engine {
    pub fn start() -> Self {
        // An engine started separately, for example with start.cmd, is left alone.
        if is_listening() {
            return Self { state: None };
        }
        let Some(root) = project_root() else {
            eprintln!("MosImage engine not started: no folder with backend/.venv found; set MOSIMAGE_ROOT");
            return Self { state: None };
        };
        let mut engine = Supervised {
            root,
            token: new_token(),
            child: None,
            started_at: Instant::now(),
            quick_failures: 0,
            stopping: false,
            #[cfg(windows)]
            job: job::Job::kill_on_close(),
        };
        if let Err(error) = engine.spawn(false) {
            eprintln!("MosImage engine did not start: {error}");
            return Self { state: None };
        }
        let state = Arc::new(Mutex::new(engine));
        let watched = Arc::clone(&state);
        thread::spawn(move || supervise(&watched));
        Self { state: Some(state) }
    }

    /// Asks the engine to cancel its jobs and unload its models, then waits for it to
    /// exit so its RAM and GPU memory are free before the app closes.
    pub fn stop(&self) {
        let Some(state) = &self.state else { return };
        let (child, token) = {
            let mut engine = lock(state);
            engine.stopping = true;
            (engine.child.take(), engine.token.clone())
        };
        if let Some(mut child) = child {
            let mut exited = false;
            if request_shutdown(&token).is_ok() {
                let deadline = Instant::now() + STOP_GRACE;
                while Instant::now() < deadline {
                    match child.try_wait() {
                        Ok(Some(_)) => {
                            exited = true;
                            break;
                        }
                        Ok(None) => thread::sleep(Duration::from_millis(100)),
                        Err(_) => break,
                    }
                }
            }
            if !exited {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        // The venv python.exe is a launcher; this also ends the interpreter it started.
        #[cfg(windows)]
        if let Some(job) = &lock(state).job {
            job.terminate();
        }
    }
}

impl Supervised {
    fn spawn(&mut self, restarted: bool) -> io::Result<()> {
        let backend = self.root.join("backend");
        let logs = self.root.join("data").join("logs");
        fs::create_dir_all(&logs)?;
        let path = logs.join("engine.log");
        let log = if restarted { OpenOptions::new().append(true).create(true).open(path)? } else { File::create(path)? };

        let mut command = Command::new(python(&backend));
        command
            .args(["-m", "uvicorn", "app.main:app", "--host", HOST, "--port", &PORT.to_string()])
            .current_dir(&backend)
            .env("MOSIMAGE_SHUTDOWN_TOKEN", &self.token)
            // Enables the hard cancel, which exits with a non-zero code and relies on a restart.
            .env("MOSIMAGE_SUPERVISED", "true")
            // macOS and Linux: the engine exits if this app dies, the job object's role on Windows.
            .env("MOSIMAGE_EXIT_WITH_PARENT", "true")
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let child = command.spawn()?;
        #[cfg(windows)]
        if let Some(job) = &self.job {
            job.assign(&child);
        }
        self.child = Some(child);
        self.started_at = Instant::now();
        Ok(())
    }
}

/// Restarts the engine after a non-zero exit, such as a hard cancel; exit code 0 means it was asked to stop.
fn supervise(state: &Mutex<Supervised>) {
    loop {
        thread::sleep(Duration::from_millis(250));
        let mut engine = lock(state);
        if engine.stopping {
            return;
        }
        let Some(child) = engine.child.as_mut() else { return };
        let status = match child.try_wait() {
            Ok(Some(status)) => status,
            Ok(None) => continue,
            Err(_) => return,
        };
        engine.child = None;
        if status.success() {
            return;
        }
        if engine.started_at.elapsed() < QUICK_FAILURE {
            engine.quick_failures += 1;
            if engine.quick_failures >= MAX_QUICK_FAILURES {
                eprintln!("MosImage engine keeps failing on start; see data/logs/engine.log");
                return;
            }
        } else {
            engine.quick_failures = 0;
        }
        drop(engine);
        thread::sleep(RESTART_DELAY);
        let mut engine = lock(state);
        if engine.stopping {
            return;
        }
        if let Err(error) = engine.spawn(true) {
            eprintln!("MosImage engine did not restart: {error}");
            return;
        }
    }
}

fn lock(state: &Mutex<Supervised>) -> MutexGuard<'_, Supervised> {
    state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// MOSIMAGE_ROOT, else the checkout this app was built or run from.
fn project_root() -> Option<PathBuf> {
    let configured = std::env::var_os("MOSIMAGE_ROOT").map(PathBuf::from);
    let beside_exe = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.ancestors().find(|dir| is_project_root(dir)).map(Path::to_path_buf));
    let built_from = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    [configured, beside_exe, Some(built_from)]
        .into_iter()
        .flatten()
        .find(|dir| is_project_root(dir))
}

fn is_project_root(dir: &Path) -> bool {
    python(&dir.join("backend")).is_file()
}

fn python(backend: &Path) -> PathBuf {
    if cfg!(windows) {
        backend.join(".venv").join("Scripts").join("python.exe")
    } else {
        backend.join(".venv").join("bin").join("python")
    }
}

const fn parse_port(text: &str) -> u16 {
    let digits = text.as_bytes();
    assert!(!digits.is_empty(), "MOSIMAGE_PORT is empty");
    let mut port: u32 = 0;
    let mut index = 0;
    while index < digits.len() {
        assert!(digits[index].is_ascii_digit(), "MOSIMAGE_PORT must be a number");
        port = port * 10 + (digits[index] - b'0') as u32;
        assert!(port <= 65535, "MOSIMAGE_PORT must be below 65536");
        index += 1;
    }
    port as u16
}

fn address() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], PORT))
}

fn is_listening() -> bool {
    TcpStream::connect_timeout(&address(), Duration::from_millis(300)).is_ok()
}

fn request_shutdown(token: &str) -> io::Result<()> {
    let mut stream = TcpStream::connect_timeout(&address(), Duration::from_secs(2))?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    write!(
        stream,
        "POST /api/shutdown HTTP/1.1\r\nHost: {HOST}:{PORT}\r\nX-MosImage-Token: {token}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )?;
    let mut reply = String::new();
    stream.read_to_string(&mut reply)?;
    if reply.starts_with("HTTP/1.1 202") {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::Other, "the engine refused to shut down"))
    }
}

/// A per-launch secret, so only this app can stop the engine it started.
fn new_token() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|time| time.as_nanos()).unwrap_or_default();
    (0..2)
        .map(|part| {
            let mut hasher = RandomState::new().build_hasher();
            hasher.write_u128(nanos);
            hasher.write_u32(std::process::id());
            hasher.write_u8(part);
            format!("{:016x}", hasher.finish())
        })
        .collect()
}

#[cfg(windows)]
mod job {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
        TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    /// Windows ends every process in the job when its last handle closes, so the
    /// engine and its GPU memory are released even if the app crashes or is killed.
    pub struct Job(HANDLE);

    // The handle is only closed once, on drop.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        pub fn kill_on_close() -> Option<Self> {
            unsafe {
                let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if handle.is_null() {
                    return None;
                }
                let job = Job(handle);
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let limited = SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    &info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION as *const _,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ) != 0;
                limited.then_some(job)
            }
        }

        /// Processes the child starts, such as the interpreter behind the venv launcher, join too.
        pub fn assign(&self, child: &Child) {
            unsafe {
                AssignProcessToJobObject(self.0, child.as_raw_handle() as HANDLE);
            }
        }

        pub fn terminate(&self) {
            unsafe {
                TerminateJobObject(self.0, 0);
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}
