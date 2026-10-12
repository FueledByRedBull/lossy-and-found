//! Double-click entry point: opens the control panel in its own window
//! without a console. The optimizer itself runs in `smart-compress`, which
//! must sit next to this program.
#![windows_subsystem = "windows"]

use std::process::Command;

fn main() {
    let engine = std::env::current_exe().ok().and_then(|path| {
        path.parent()
            .map(|dir| dir.join(format!("smart-compress{}", std::env::consts::EXE_SUFFIX)))
    });
    let Some(engine) = engine.filter(|path| path.is_file()) else {
        fail("smart-compress must be in the same folder as this program.");
        return;
    };
    let mut command = Command::new(engine);
    command
        .args(["--gui", "--gui-window"])
        .args(std::env::args_os().skip(1));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: the engine is a console program.
        command.creation_flags(0x0800_0000);
    }
    if let Err(error) = command.spawn() {
        fail(&format!("Could not start smart-compress: {error}"));
    }
}

#[cfg(windows)]
fn fail(message: &str) {
    #[link(name = "user32")]
    unsafe extern "system" {
        fn MessageBoxW(
            window: *mut std::ffi::c_void,
            text: *const u16,
            caption: *const u16,
            kind: u32,
        ) -> i32;
    }
    let wide = |text: &str| text.encode_utf16().chain([0]).collect::<Vec<u16>>();
    let (text, caption) = (wide(message), wide("smart-compress"));
    // MB_ICONERROR
    unsafe {
        MessageBoxW(std::ptr::null_mut(), text.as_ptr(), caption.as_ptr(), 0x10);
    }
}

#[cfg(not(windows))]
fn fail(message: &str) {
    eprintln!("{message}");
}
