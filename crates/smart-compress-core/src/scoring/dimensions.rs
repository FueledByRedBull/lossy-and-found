use image::metadata::Orientation;

/// Largest image, in pixels, the metrics read as it is. The built-in metrics
/// reduce a larger one to at most 2048 px on a side; the external ones read a
/// mosaic of samples from it.
pub const FULL_SIZE_METRIC_PIXELS: u64 = 16_000_000;

pub(crate) fn bounded_metric_dimensions(width: u32, height: u32) -> (u32, u32) {
    let pixels = width as u64 * height as u64;
    if pixels <= FULL_SIZE_METRIC_PIXELS || width == 0 || height == 0 {
        return (width, height);
    }
    let longest = width.max(height) as f64;
    let scale = 2048.0 / longest;
    (
        ((width as f64 * scale).round() as u32).max(1),
        ((height as f64 * scale).round() as u32).max(1),
    )
}

pub(super) fn bounded_preview_dimensions(width: u32, height: u32, max_side: u32) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (width, height);
    }
    let longest = width.max(height);
    if longest <= max_side {
        return (width, height);
    }
    let scale = max_side as f64 / longest as f64;
    (
        ((width as f64 * scale).round() as u32).max(1),
        ((height as f64 * scale).round() as u32).max(1),
    )
}

pub(super) fn oriented_dimensions(width: u32, height: u32, orientation: Orientation) -> (u32, u32) {
    match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Rotate90FlipH
        | Orientation::Rotate270FlipH => (height, width),
        _ => (width, height),
    }
}
