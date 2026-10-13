use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::{process::run_process, types::Codec};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    /// Where the tool runs from; reports and the control panel do not carry it.
    #[serde(skip_serializing)]
    pub path: Option<PathBuf>,
    pub version: Option<String>,
    pub enabled: bool,
    pub install_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInventory {
    pub cwebp: ToolInfo,
    pub webpmux: ToolInfo,
    pub avifenc: ToolInfo,
    pub avifdec: ToolInfo,
    pub cjxl: ToolInfo,
    pub djxl: ToolInfo,
    pub oxipng: ToolInfo,
    pub pngquant: ToolInfo,
    pub cjpegli: ToolInfo,
    pub ssimulacra2: ToolInfo,
}

impl ToolInventory {
    /// Detection spawns every tool once; the result is reused for the process lifetime.
    pub fn detect() -> Self {
        static INVENTORY: OnceLock<ToolInventory> = OnceLock::new();
        INVENTORY.get_or_init(Self::detect_uncached).clone()
    }

    fn detect_uncached() -> Self {
        Self {
            cwebp: detect_tool(
                "cwebp",
                &["-version"],
                "Install libwebp tools or place cwebp.exe under tools/libwebp/.../bin.",
            ),
            webpmux: detect_tool(
                "webpmux",
                &["-version"],
                "Install libwebp tools or place webpmux.exe next to cwebp.exe; WebP metadata needs it.",
            ),
            avifenc: detect_tool(
                "avifenc",
                &["--version"],
                "Install libavif tools or place avifenc.exe and avifdec.exe under tools/libavif.",
            ),
            avifdec: detect_tool(
                "avifdec",
                &["--version"],
                "Install libavif tools or place avifenc.exe and avifdec.exe under tools/libavif.",
            ),
            cjxl: detect_tool(
                "cjxl",
                &["--version"],
                "Install JPEG XL tools or place cjxl.exe and djxl.exe under tools/libjxl/bin.",
            ),
            djxl: detect_tool(
                "djxl",
                &["--version"],
                "Install JPEG XL tools or place cjxl.exe and djxl.exe under tools/libjxl/bin.",
            ),
            oxipng: detect_tool(
                "oxipng",
                &["--version"],
                "Install oxipng, for example with winget, cargo install, or a project-local binary.",
            ),
            pngquant: detect_tool(
                "pngquant",
                &["--version"],
                "Install pngquant or place pngquant.exe under tools/pngquant/pngquant.",
            ),
            cjpegli: detect_tool(
                "cjpegli",
                &["--version"],
                "Install JPEG XL/jpegli tools or place cjpegli.exe under tools/libjxl/bin.",
            ),
            ssimulacra2: detect_tool(
                "ssimulacra2",
                &["--help"],
                "Install JPEG XL tools or place ssimulacra2.exe under tools/libjxl/bin.",
            ),
        }
    }

    /// Tools a format needs that were not found: its encoder, and the decoder
    /// that reads a candidate back to score it.
    pub fn missing_for(&self, codec: Codec) -> Vec<&str> {
        let needed: &[&ToolInfo] = match codec {
            Codec::Png | Codec::Jpeg => &[],
            Codec::Webp => &[&self.cwebp],
            Codec::Avif => &[&self.avifenc, &self.avifdec],
            Codec::Jxl => &[&self.cjxl, &self.djxl],
        };
        needed
            .iter()
            .filter(|tool| !tool.enabled)
            .map(|tool| tool.name.as_str())
            .collect()
    }

    pub fn available_for(&self, codec: Codec) -> bool {
        self.missing_for(codec).is_empty()
    }

    fn tool_named(&self, name: &str) -> Option<&ToolInfo> {
        [
            &self.cwebp,
            &self.webpmux,
            &self.avifenc,
            &self.avifdec,
            &self.cjxl,
            &self.djxl,
            &self.oxipng,
            &self.pngquant,
            &self.cjpegli,
            &self.ssimulacra2,
        ]
        .into_iter()
        .find(|tool| tool.name == name)
    }
}

/// Setting this variable turns every skipped test that needs an external tool
/// into a failure.
pub(crate) const REQUIRE_TOOLS_ENV: &str = "SMART_COMPRESS_REQUIRE_TOOLS";

/// Test support shared by the unit and integration tests: the tool inventory,
/// if every tool named in `needed` (by the name the inventory gives it, such
/// as `cwebp` or `ssimulacra2`) is installed. Otherwise the test skips
/// with one line on stderr naming what is missing, unless
/// `SMART_COMPRESS_REQUIRE_TOOLS` is set, which fails it.
#[doc(hidden)]
pub fn tools_or_skip(needed: &[&str]) -> Option<ToolInventory> {
    tools_or_skip_with(
        ToolInventory::detect(),
        needed,
        std::env::var_os(REQUIRE_TOOLS_ENV).is_some(),
    )
}

pub(crate) fn tools_or_skip_with(
    tools: ToolInventory,
    needed: &[&str],
    required: bool,
) -> Option<ToolInventory> {
    let missing = needed
        .iter()
        .filter(|name| {
            !tools
                .tool_named(name)
                .unwrap_or_else(|| panic!("no tool is named {name}"))
                .enabled
        })
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    if missing.is_empty() {
        return Some(tools);
    }
    let test = std::thread::current().name().unwrap_or("test").to_owned();
    assert!(
        !required,
        "{test}: missing {missing} and {REQUIRE_TOOLS_ENV} is set"
    );
    eprintln!("{test}: skipped, missing {missing}; set {REQUIRE_TOOLS_ENV} to fail instead");
    None
}

/// The tools `tools_or_skip` should be asked for to run `codecs`: their
/// encoders, and the decoders that read a candidate back.
#[doc(hidden)]
pub fn tool_names_for(codecs: &[Codec]) -> Vec<&'static str> {
    codecs
        .iter()
        .flat_map(|codec| match codec {
            Codec::Png | Codec::Jpeg => &[][..],
            Codec::Webp => &["cwebp"][..],
            Codec::Avif => &["avifenc", "avifdec"][..],
            Codec::Jxl => &["cjxl", "djxl"][..],
        })
        .copied()
        .collect()
}

fn detect_tool(name: &str, version_args: &[&str], install_hint: &str) -> ToolInfo {
    let path = find_in_path(name);
    let version = path.as_ref().and_then(|tool_path| {
        let args = version_args.iter().map(OsString::from).collect::<Vec<_>>();
        run_process(tool_path, &args, Duration::from_secs(5))
            .map(|output| {
                let text = if output.stdout.is_empty() {
                    String::from_utf8_lossy(&output.stderr).to_string()
                } else {
                    String::from_utf8_lossy(&output.stdout).to_string()
                };
                text.lines()
                    .next()
                    .unwrap_or("version unavailable")
                    .to_string()
            })
            .ok()
    });

    let enabled = path.is_some();
    ToolInfo {
        name: name.to_string(),
        enabled,
        path,
        version,
        install_hint: (!enabled).then(|| install_hint.to_string()),
    }
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let extensions: Vec<String> = if cfg!(windows) {
        std::env::var_os("PATHEXT")
            .map(|value| {
                std::env::split_paths(&value)
                    .filter_map(|path| path.to_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_else(|| vec![".EXE".to_string(), ".BAT".to_string(), ".CMD".to_string()])
    } else {
        vec!["".to_string()]
    };

    let mut search_dirs = std::env::var_os("PATH")
        .map(|path_var| absolute_path_dirs(&path_var))
        .unwrap_or_default();
    search_dirs.extend(project_tool_dirs());

    for dir in search_dirs {
        let direct = dir.join(name);
        if direct.is_file() {
            return Some(direct);
        }
        for ext in &extensions {
            let candidate = dir.join(format!("{name}{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// The absolute entries of a PATH value. An empty or relative entry means the
/// working directory, which would run whatever `cjxl.exe` an untrusted folder
/// happened to contain.
pub(crate) fn absolute_path_dirs(path_var: &OsStr) -> Vec<PathBuf> {
    std::env::split_paths(path_var)
        .filter(|dir| dir.is_absolute())
        .collect()
}

/// Tool folders bundled with the program: those beside the executable and,
/// for a build inside this workspace, those of the workspace. Nothing else
/// above the executable is searched, since any user can create a `tools` folder
/// in the root of a drive.
fn project_tool_dirs() -> Vec<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(tool_dirs_for))
        .unwrap_or_default()
}

pub(crate) fn tool_dirs_for(exe_dir: &Path) -> Vec<PathBuf> {
    let mut roots = vec![exe_dir];
    // A development build lives in `<workspace>/target/...`.
    if let Some(workspace) = exe_dir.ancestors().find(|dir| {
        dir.join("Cargo.toml").is_file()
            && dir.join("tools").is_dir()
            && exe_dir
                .strip_prefix(dir)
                .is_ok_and(|rest| rest.starts_with("target"))
    }) {
        roots.push(workspace);
    }

    let mut dirs = Vec::new();
    for root in roots {
        for relative in [
            "tools",
            "tools/bin",
            "tools/libjxl/bin",
            "tools/libavif",
            "tools/libwebp/libwebp-1.6.0-windows-x64/bin",
            "tools/pngquant/pngquant",
        ] {
            let dir = root.join(relative);
            if dir.is_dir() && !dirs.contains(&dir) {
                dirs.push(dir);
            }
        }
    }
    dirs
}
