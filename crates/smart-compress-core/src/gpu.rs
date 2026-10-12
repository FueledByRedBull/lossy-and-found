use serde::{Deserialize, Serialize};

use crate::{config::GpuConfig, scoring::bounded_metric_dimensions};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum GpuStatus {
    Disabled,
    Enabled,
    Unavailable,
    RequiredUnavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuReport {
    pub status: GpuStatus,
    pub requested_backend: String,
    pub requested_adapter_name: Option<String>,
    pub requested_adapter_vendor: Option<u32>,
    pub requested_adapter_device: Option<u32>,
    pub selected_adapter: Option<GpuAdapterReport>,
    pub available_adapters: Vec<GpuAdapterReport>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAdapterReport {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub backend: String,
    pub max_texture_dimension_2d: u32,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    /// Whether the adapter supports timestamp queries for GPU-side pass timing.
    #[serde(default)]
    pub timestamp_queries: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuWorkEstimate {
    pub source_width: u32,
    pub source_height: u32,
    pub source_pixels: u64,
    pub metric_width: u32,
    pub metric_height: u32,
    pub metric_pixels: u64,
    pub source_rgba_bytes: u64,
    pub metric_rgba_bytes: u64,
    pub metric_upload_bytes_per_candidate: u64,
    pub metric_readback_bytes_per_candidate: u64,
    pub estimated_peak_metric_bytes_per_candidate: u64,
    pub limiting_bytes: Option<u64>,
    pub metric_tiling_required: bool,
    pub selected_metric_tile_pixels: Option<u64>,
    pub tiling_required_for_full_source_preprocess: bool,
    pub selected_tile_pixels: Option<u64>,
}

impl GpuWorkEstimate {
    pub fn for_image(
        width: u32,
        height: u32,
        configured_max_vram_bytes: Option<u64>,
        adapter: Option<&GpuAdapterReport>,
    ) -> Self {
        let source_pixels = width as u64 * height as u64;
        let (metric_width, metric_height) = bounded_metric_dimensions(width, height);
        let metric_pixels = metric_width as u64 * metric_height as u64;
        let source_rgba_bytes = rgba_bytes(source_pixels);
        let metric_rgba_bytes = rgba_bytes(metric_pixels);
        let metric_partials_bytes = metric_readback_bytes(metric_pixels);
        let metric_upload_bytes_per_candidate = metric_rgba_bytes.saturating_mul(2);
        let metric_readback_bytes_per_candidate = metric_partials_bytes;
        let estimated_peak_metric_bytes_per_candidate = metric_upload_bytes_per_candidate
            .saturating_add(metric_partials_bytes.saturating_mul(2))
            .saturating_add(16);
        let limiting_bytes = gpu_limit_bytes(configured_max_vram_bytes, adapter);
        let metric_tiling_required = limiting_bytes
            .is_some_and(|limit| metric_dispatch_bytes(metric_width, metric_height) > limit);
        let selected_metric_tile_pixels = metric_tiling_required
            .then(|| {
                selected_metric_tile_pixels(metric_width, metric_height, limiting_bytes.unwrap())
            })
            .flatten();
        let tiling_required_for_full_source_preprocess = limiting_bytes.is_some_and(|limit| {
            full_source_preprocess_peak_bytes(source_rgba_bytes) > limit
                || adapter.is_some_and(|adapter| {
                    source_rgba_bytes > adapter.max_buffer_size
                        || source_rgba_bytes > adapter.max_storage_buffer_binding_size
                })
        });
        let selected_tile_pixels = tiling_required_for_full_source_preprocess
            .then(|| selected_tile_pixels(limiting_bytes.unwrap_or(source_rgba_bytes)));

        Self {
            source_width: width,
            source_height: height,
            source_pixels,
            metric_width,
            metric_height,
            metric_pixels,
            source_rgba_bytes,
            metric_rgba_bytes,
            metric_upload_bytes_per_candidate,
            metric_readback_bytes_per_candidate,
            estimated_peak_metric_bytes_per_candidate,
            limiting_bytes,
            metric_tiling_required,
            selected_metric_tile_pixels,
            tiling_required_for_full_source_preprocess,
            selected_tile_pixels,
        }
    }
}

pub fn detect_gpu(config: &GpuConfig) -> GpuReport {
    detect_gpu_with_adapter(config).0
}

/// Enumerates adapters once and returns the report together with the exact
/// adapter it names as selected, so a device can be created from that adapter.
/// The adapter is `None` whenever `selected_adapter` is.
pub fn detect_gpu_with_adapter(config: &GpuConfig) -> (GpuReport, Option<wgpu::Adapter>) {
    if !config.enabled {
        return (
            GpuReport {
                status: GpuStatus::Disabled,
                requested_backend: config.backend.clone(),
                requested_adapter_name: config.adapter_name.clone(),
                requested_adapter_vendor: config.adapter_vendor,
                requested_adapter_device: config.adapter_device,
                selected_adapter: None,
                available_adapters: Vec::new(),
                warnings: vec!["GPU disabled by configuration".to_string()],
            },
            None,
        );
    }

    if config.backend != "wgpu" {
        return (
            unavailable(
                config,
                format!(
                    "unsupported GPU backend '{}'; only 'wgpu' is implemented",
                    config.backend
                ),
            ),
            None,
        );
    }

    let mut candidates = select_adapters(
        config.adapter_name.as_deref(),
        config.adapter_vendor,
        config.adapter_device,
    )
    .into_iter();
    let (selected, selected_adapter) = match candidates.next() {
        Some((adapter, report)) => (Some(adapter), Some(report)),
        None => (None, None),
    };
    let adapters: Vec<GpuAdapterReport> = selected_adapter
        .iter()
        .cloned()
        .chain(candidates.map(|(_, report)| report))
        .collect();
    let status = if selected_adapter.is_some() {
        GpuStatus::Enabled
    } else if config.required {
        GpuStatus::RequiredUnavailable
    } else {
        GpuStatus::Unavailable
    };
    let warnings = if selected_adapter.is_some() {
        Vec::new()
    } else {
        vec![if config.adapter_name.is_some()
            || config.adapter_vendor.is_some()
            || config.adapter_device.is_some()
        {
            "No GPU adapter matched the requested selector".to_string()
        } else {
            "No hardware GPU adapter found through wgpu; software adapters are used only when named with --gpu-adapter-name".to_string()
        }]
    };

    (
        GpuReport {
            status,
            requested_backend: config.backend.clone(),
            requested_adapter_name: config.adapter_name.clone(),
            requested_adapter_vendor: config.adapter_vendor,
            requested_adapter_device: config.adapter_device,
            selected_adapter,
            available_adapters: adapters,
            warnings,
        },
        selected,
    )
}

fn unavailable(config: &GpuConfig, warning: String) -> GpuReport {
    GpuReport {
        status: if config.required {
            GpuStatus::RequiredUnavailable
        } else {
            GpuStatus::Unavailable
        },
        requested_backend: config.backend.clone(),
        requested_adapter_name: config.adapter_name.clone(),
        requested_adapter_vendor: config.adapter_vendor,
        requested_adapter_device: config.adapter_device,
        selected_adapter: None,
        available_adapters: Vec::new(),
        warnings: vec![warning],
    }
}

/// The one place that decides which adapter is used: enumerates once and
/// returns the adapters matching the selector, best first.
pub(crate) fn select_adapters(
    name: Option<&str>,
    vendor: Option<u32>,
    device: Option<u32>,
) -> Vec<(wgpu::Adapter, GpuAdapterReport)> {
    let instance = wgpu::Instance::default();
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .map(|adapter| {
            let report = adapter_report(&adapter);
            (adapter, report)
        })
        .collect();
    filter_and_rank(adapters, name, vendor, device)
}

pub(crate) fn adapter_report(adapter: &wgpu::Adapter) -> GpuAdapterReport {
    let info = adapter.get_info();
    let limits = adapter.limits();
    GpuAdapterReport {
        name: info.name,
        vendor: info.vendor,
        device: info.device,
        device_type: format!("{:?}", info.device_type),
        backend: format!("{:?}", info.backend),
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
        max_buffer_size: limits.max_buffer_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        timestamp_queries: adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
    }
}

/// Keeps entries whose report matches every given selector (name is a
/// case-insensitive substring) and orders them by device type, stable.
///
/// A software adapter (device type `Cpu`, such as Windows' Basic Render
/// Driver) is left out unless a selector is given: scoring on a software
/// rasteriser is slower than the CPU backend, and a machine without a GPU
/// should not report that it has one. Naming it selects it.
fn filter_and_rank<T>(
    mut adapters: Vec<(T, GpuAdapterReport)>,
    name: Option<&str>,
    vendor: Option<u32>,
    device: Option<u32>,
) -> Vec<(T, GpuAdapterReport)> {
    let automatic = name.is_none() && vendor.is_none() && device.is_none();
    adapters.retain(|(_, report)| {
        (!automatic || report.device_type != SOFTWARE_DEVICE_TYPE)
            && name.is_none_or(|name| {
                report
                    .name
                    .to_ascii_lowercase()
                    .contains(&name.to_ascii_lowercase())
            })
            && vendor.is_none_or(|vendor| report.vendor == vendor)
            && device.is_none_or(|device| report.device == device)
    });
    adapters.sort_by_key(|(_, report)| adapter_rank(&report.device_type));
    adapters
}

/// The device type wgpu gives to software adapters.
pub(crate) const SOFTWARE_DEVICE_TYPE: &str = "Cpu";

fn adapter_rank(device_type: &str) -> u8 {
    match device_type {
        "DiscreteGpu" => 0,
        "IntegratedGpu" => 1,
        "VirtualGpu" => 2,
        SOFTWARE_DEVICE_TYPE => 3,
        _ => 4,
    }
}

fn rgba_bytes(pixels: u64) -> u64 {
    pixels.saturating_mul(4)
}

fn metric_readback_bytes(metric_pixels: u64) -> u64 {
    const COMPONENTS: u64 = 8;
    const WORKGROUP_SIZE: u64 = 256;
    metric_pixels
        .div_ceil(WORKGROUP_SIZE)
        .saturating_mul(COMPONENTS)
        .saturating_mul(std::mem::size_of::<u32>() as u64)
}

fn gpu_limit_bytes(
    configured_max_vram_bytes: Option<u64>,
    adapter: Option<&GpuAdapterReport>,
) -> Option<u64> {
    [
        configured_max_vram_bytes,
        adapter.map(|adapter| adapter.max_buffer_size),
        adapter.map(|adapter| adapter.max_storage_buffer_binding_size),
    ]
    .into_iter()
    .flatten()
    .min()
}

fn full_source_preprocess_peak_bytes(source_rgba_bytes: u64) -> u64 {
    source_rgba_bytes.saturating_mul(2)
}

fn selected_tile_pixels(limit_bytes: u64) -> u64 {
    let usable_bytes = limit_bytes / 2;
    (usable_bytes / 4).max(1)
}

fn selected_metric_tile_pixels(width: u32, height: u32, limit_bytes: u64) -> Option<u64> {
    let mut low = 1_u32;
    let mut high = height;
    let mut best = 0_u32;
    while low <= high {
        let mid = low + (high - low) / 2;
        let buffer_rows = mid.saturating_add(2).min(height);
        if metric_dispatch_bytes(width, buffer_rows) <= limit_bytes {
            best = mid;
            low = mid.saturating_add(1);
        } else {
            high = mid.saturating_sub(1);
        }
    }
    (best > 0).then_some(width as u64 * best as u64)
}

fn metric_dispatch_bytes(width: u32, rows: u32) -> u64 {
    const COMPONENTS: u64 = 8;
    const WORKGROUP_SIZE: u64 = 256;
    let pixels = width as u64 * rows as u64;
    let rgba_bytes = pixels.saturating_mul(4);
    let partial_bytes = pixels
        .div_ceil(WORKGROUP_SIZE)
        .saturating_mul(COMPONENTS)
        .saturating_mul(std::mem::size_of::<u32>() as u64);
    rgba_bytes
        .saturating_mul(2)
        .saturating_add(partial_bytes.saturating_mul(2))
        .saturating_add(32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake(name: &str, device_type: &str, backend: &str) -> ((), GpuAdapterReport) {
        (
            (),
            GpuAdapterReport {
                name: name.to_string(),
                vendor: 1,
                device: 2,
                device_type: device_type.to_string(),
                backend: backend.to_string(),
                max_texture_dimension_2d: 0,
                max_buffer_size: 0,
                max_storage_buffer_binding_size: 0,
                timestamp_queries: false,
            },
        )
    }

    fn names(adapters: &[((), GpuAdapterReport)]) -> Vec<&str> {
        adapters.iter().map(|(_, a)| a.name.as_str()).collect()
    }

    fn sample() -> Vec<((), GpuAdapterReport)> {
        vec![
            fake("Basic Render Driver", "Cpu", "Dx12"),
            fake("AMD Radeon", "DiscreteGpu", "Vulkan"),
            fake("AMD Radeon", "DiscreteGpu", "Dx12"),
            fake("Intel UHD", "IntegratedGpu", "Vulkan"),
        ]
    }

    #[test]
    fn default_selection_prefers_discrete() {
        let ranked = filter_and_rank(sample(), None, None, None);
        assert_eq!(ranked[0].1.device_type, "DiscreteGpu");
        assert_eq!(ranked[0].1.backend, "Vulkan", "ties keep enumeration order");
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[2].1.device_type, "IntegratedGpu");
    }

    #[test]
    fn a_software_adapter_is_not_selected_automatically() {
        let ranked = filter_and_rank(sample(), None, None, None);
        assert_eq!(ranked.len(), 3);
        assert!(ranked.iter().all(|(_, report)| report.device_type != "Cpu"));

        // A machine with nothing but the software adapter has no GPU.
        let only_software = vec![fake("Basic Render Driver", "Cpu", "Dx12")];
        assert!(filter_and_rank(only_software, None, None, None).is_empty());
    }

    #[test]
    fn a_software_adapter_named_explicitly_is_still_selected() {
        let named = filter_and_rank(sample(), Some("basic render"), None, None);
        assert_eq!(names(&named), ["Basic Render Driver"]);
        let only_software = vec![fake("Basic Render Driver", "Cpu", "Dx12")];
        assert_eq!(filter_and_rank(only_software, None, Some(1), None).len(), 1);
    }

    #[test]
    fn selection_by_name_is_case_insensitive_substring() {
        let ranked = filter_and_rank(sample(), Some("basic render"), None, None);
        assert_eq!(names(&ranked), ["Basic Render Driver"]);
    }

    #[test]
    fn selection_by_vendor_and_device_ids() {
        assert_eq!(filter_and_rank(sample(), None, Some(1), Some(2)).len(), 4);
        assert_eq!(filter_and_rank(sample(), None, Some(1), None).len(), 4);
        assert!(filter_and_rank(sample(), None, Some(1), Some(3)).is_empty());
        assert!(filter_and_rank(sample(), None, Some(9), None).is_empty());
    }

    #[test]
    fn selection_with_no_match_is_empty() {
        assert!(filter_and_rank(sample(), Some("nvidia"), None, None).is_empty());
    }

    #[test]
    fn gpu_work_estimate_bounds_large_metric_surface() {
        let estimate = GpuWorkEstimate::for_image(16_304, 12_256, None, None);

        assert_eq!(estimate.source_pixels, 199_821_824);
        assert!(estimate.metric_width <= 2048);
        assert!(estimate.metric_height <= 2048);
        assert!(estimate.metric_pixels < estimate.source_pixels);
        assert!(estimate.estimated_peak_metric_bytes_per_candidate > 0);
        assert!(!estimate.metric_tiling_required);
    }

    #[test]
    fn gpu_work_estimate_marks_full_source_tiling_when_limit_is_tight() {
        let estimate = GpuWorkEstimate::for_image(16_304, 12_256, Some(128 * 1024 * 1024), None);

        assert!(estimate.tiling_required_for_full_source_preprocess);
        assert!(estimate.selected_tile_pixels.is_some());
    }

    #[test]
    fn gpu_work_estimate_marks_metric_tiling_when_metric_limit_is_tight() {
        let estimate = GpuWorkEstimate::for_image(2048, 2048, Some(2 * 1024 * 1024), None);

        assert!(estimate.metric_tiling_required);
        assert!(estimate.selected_metric_tile_pixels.is_some());
    }
}
