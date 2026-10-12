pub(super) const METRIC_SUMS_SHADER: &str = r#"
struct Params {
    pixel_count: u32,
    width: u32,
    buffer_height: u32,
    global_y_offset: u32,
    global_height: u32,
    score_start_local_y: u32,
    score_row_count: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> original: array<u32>;
@group(0) @binding(1) var<storage, read> candidate: array<u32>;
@group(0) @binding(2) var<storage, read_write> partials: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> scratch: array<u32, 256>;
var<workgroup> scratch_alpha: array<u32, 256>;
var<workgroup> scratch_comp0: array<u32, 256>;
var<workgroup> scratch_comp1: array<u32, 256>;
var<workgroup> scratch_comp2: array<u32, 256>;
var<workgroup> scratch_comp3: array<u32, 256>;
var<workgroup> scratch_comp4: array<u32, 256>;
var<workgroup> scratch_edge: array<u32, 256>;
var<workgroup> scratch_visible: array<u32, 256>;
var<workgroup> scratch_visible_edge: array<u32, 256>;

fn channel(value: u32, shift: u32) -> i32 {
    return i32((value >> shift) & 0xffu);
}

fn sqdiff(a: i32, b: i32) -> u32 {
    let diff = a - b;
    return u32(diff * diff);
}

fn blend_channel(foreground: i32, alpha: i32, background: i32) -> i32 {
    return (foreground * alpha + background * (255i - alpha) + 127i) / 255i;
}

fn luma(value: u32) -> i32 {
    let r = f32(channel(value, 0u));
    let g = f32(channel(value, 8u));
    let b = f32(channel(value, 16u));
    return i32(round(0.2126 * r + 0.7152 * g + 0.0722 * b));
}

// Fully transparent in both images: its colour is invisible, so it is no sample
// of the raw RGB and edge scores and has luma 0 in both.
fn hidden(image_index: u32) -> bool {
    return channel(original[image_index], 24u) == 0 && channel(candidate[image_index], 24u) == 0;
}

fn luma_original(image_index: u32) -> i32 {
    if (hidden(image_index)) {
        return 0;
    }
    return luma(original[image_index]);
}

fn luma_candidate(image_index: u32) -> i32 {
    if (hidden(image_index)) {
        return 0;
    }
    return luma(candidate[image_index]);
}

fn edge_original(image_index: u32) -> i32 {
    let left = luma_original(image_index - 1u);
    let right = luma_original(image_index + 1u);
    let top = luma_original(image_index - params.width);
    let bottom = luma_original(image_index + params.width);
    return abs(left - right) + abs(top - bottom);
}

fn edge_candidate(image_index: u32) -> i32 {
    let left = luma_candidate(image_index - 1u);
    let right = luma_candidate(image_index + 1u);
    let top = luma_candidate(image_index - params.width);
    let bottom = luma_candidate(image_index + params.width);
    return abs(left - right) + abs(top - bottom);
}

fn composite_sse(a: u32, b: u32, background: vec3<i32>) -> u32 {
    let aa = channel(a, 24u);
    let ba = channel(b, 24u);
    let ar = blend_channel(channel(a, 0u), aa, background.x);
    let ag = blend_channel(channel(a, 8u), aa, background.y);
    let ab = blend_channel(channel(a, 16u), aa, background.z);
    let br = blend_channel(channel(b, 0u), ba, background.x);
    let bg = blend_channel(channel(b, 8u), ba, background.y);
    let bb = blend_channel(channel(b, 16u), ba, background.z);
    return sqdiff(ar, br) + sqdiff(ag, bg) + sqdiff(ab, bb);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let index = global_id.x;
    var sum_rgb: u32 = 0u;
    var sum_alpha: u32 = 0u;
    var sum_comp0: u32 = 0u;
    var sum_comp1: u32 = 0u;
    var sum_comp2: u32 = 0u;
    var sum_comp3: u32 = 0u;
    var sum_comp4: u32 = 0u;
    var sum_edge: u32 = 0u;
    var sum_visible: u32 = 0u;
    var sum_visible_edge: u32 = 0u;
    if (index < params.pixel_count) {
        let x = index % params.width;
        let y = index / params.width;
        let global_y = params.global_y_offset + y;
        let scored = y >= params.score_start_local_y
            && y < params.score_start_local_y + params.score_row_count;
        if (scored) {
            let a = original[index];
            let b = candidate[index];
            let visible = !hidden(index);
            if (visible) {
                sum_rgb = sqdiff(channel(a, 0u), channel(b, 0u))
                    + sqdiff(channel(a, 8u), channel(b, 8u))
                    + sqdiff(channel(a, 16u), channel(b, 16u));
                sum_visible = 1u;
            }
            sum_alpha = sqdiff(channel(a, 24u), channel(b, 24u));
            sum_comp0 = composite_sse(a, b, vec3<i32>(255i, 255i, 255i));
            sum_comp1 = composite_sse(a, b, vec3<i32>(0i, 0i, 0i));
            sum_comp2 = composite_sse(a, b, vec3<i32>(128i, 128i, 128i));
            sum_comp3 = composite_sse(a, b, vec3<i32>(238i, 238i, 238i));
            sum_comp4 = composite_sse(a, b, vec3<i32>(64i, 64i, 64i));
            if (visible
                && x > 0u
                && y > 0u
                && global_y > 0u
                && x + 1u < params.width
                && y + 1u < params.buffer_height
                && global_y + 1u < params.global_height) {
                sum_edge = sqdiff(edge_original(index), edge_candidate(index));
                sum_visible_edge = 1u;
            }
        }
    }
    scratch[local_id.x] = sum_rgb;
    scratch_alpha[local_id.x] = sum_alpha;
    scratch_comp0[local_id.x] = sum_comp0;
    scratch_comp1[local_id.x] = sum_comp1;
    scratch_comp2[local_id.x] = sum_comp2;
    scratch_comp3[local_id.x] = sum_comp3;
    scratch_comp4[local_id.x] = sum_comp4;
    scratch_edge[local_id.x] = sum_edge;
    scratch_visible[local_id.x] = sum_visible;
    scratch_visible_edge[local_id.x] = sum_visible_edge;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (local_id.x < stride) {
            scratch[local_id.x] = scratch[local_id.x] + scratch[local_id.x + stride];
            scratch_alpha[local_id.x] = scratch_alpha[local_id.x] + scratch_alpha[local_id.x + stride];
            scratch_comp0[local_id.x] = scratch_comp0[local_id.x] + scratch_comp0[local_id.x + stride];
            scratch_comp1[local_id.x] = scratch_comp1[local_id.x] + scratch_comp1[local_id.x + stride];
            scratch_comp2[local_id.x] = scratch_comp2[local_id.x] + scratch_comp2[local_id.x + stride];
            scratch_comp3[local_id.x] = scratch_comp3[local_id.x] + scratch_comp3[local_id.x + stride];
            scratch_comp4[local_id.x] = scratch_comp4[local_id.x] + scratch_comp4[local_id.x + stride];
            scratch_edge[local_id.x] = scratch_edge[local_id.x] + scratch_edge[local_id.x + stride];
            scratch_visible[local_id.x] = scratch_visible[local_id.x] + scratch_visible[local_id.x + stride];
            scratch_visible_edge[local_id.x] = scratch_visible_edge[local_id.x] + scratch_visible_edge[local_id.x + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (local_id.x == 0u) {
        let base = workgroup_id.x * 10u;
        partials[base] = scratch[0];
        partials[base + 1u] = scratch_alpha[0];
        partials[base + 2u] = scratch_comp0[0];
        partials[base + 3u] = scratch_comp1[0];
        partials[base + 4u] = scratch_comp2[0];
        partials[base + 5u] = scratch_comp3[0];
        partials[base + 6u] = scratch_comp4[0];
        partials[base + 7u] = scratch_edge[0];
        partials[base + 8u] = scratch_visible[0];
        partials[base + 9u] = scratch_visible_edge[0];
    }
}
"#;

pub(super) const TRANSFORM_DOWNSAMPLE_SHADER: &str = r#"
struct Params {
    source_width: u32,
    source_height: u32,
    output_width: u32,
    output_height: u32,
    orientation_code: u32,
    tile_x: u32,
    tile_y: u32,
    tile_width: u32,
    out_row_offset: u32,
    out_row_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn channel(value: u32, shift: u32) -> u32 {
    return (value >> shift) & 0xffu;
}

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

fn oriented_width() -> u32 {
    if (params.orientation_code >= 5u) {
        return params.source_height;
    }
    return params.source_width;
}

fn oriented_height() -> u32 {
    if (params.orientation_code >= 5u) {
        return params.source_width;
    }
    return params.source_height;
}

fn source_index(oriented_x: u32, oriented_y: u32) -> u32 {
    var source_x = oriented_x;
    var source_y = oriented_y;
    switch params.orientation_code {
        case 1u: {
        }
        case 2u: {
            source_x = params.source_width - 1u - oriented_x;
        }
        case 3u: {
            source_x = params.source_width - 1u - oriented_x;
            source_y = params.source_height - 1u - oriented_y;
        }
        case 4u: {
            source_y = params.source_height - 1u - oriented_y;
        }
        case 5u: {
            source_x = oriented_y;
            source_y = oriented_x;
        }
        case 6u: {
            source_x = oriented_y;
            source_y = params.source_height - 1u - oriented_x;
        }
        case 7u: {
            source_x = params.source_width - 1u - oriented_y;
            source_y = params.source_height - 1u - oriented_x;
        }
        default: {
            source_x = params.source_width - 1u - oriented_y;
            source_y = oriented_x;
        }
    }
    return (source_y - params.tile_y) * params.tile_width + (source_x - params.tile_x);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y + params.out_row_offset;
    if (out_x >= params.output_width || global_id.y >= params.out_row_count) {
        return;
    }

    let src_oriented_width = oriented_width();
    let src_oriented_height = oriented_height();
    let start_x = (out_x * src_oriented_width) / params.output_width;
    let start_y = (out_y * src_oriented_height) / params.output_height;
    let end_x = max(
        start_x + 1u,
        ((out_x + 1u) * src_oriented_width + params.output_width - 1u) / params.output_width,
    );
    let end_y = max(
        start_y + 1u,
        ((out_y + 1u) * src_oriented_height + params.output_height - 1u) / params.output_height,
    );

    var sum_r = 0u;
    var sum_g = 0u;
    var sum_b = 0u;
    var sum_a = 0u;
    var count = 0u;
    for (var source_y = start_y; source_y < end_y; source_y = source_y + 1u) {
        for (var source_x = start_x; source_x < end_x; source_x = source_x + 1u) {
            let pixel = source[source_index(source_x, source_y)];
            sum_r = sum_r + channel(pixel, 0u);
            sum_g = sum_g + channel(pixel, 8u);
            sum_b = sum_b + channel(pixel, 16u);
            sum_a = sum_a + channel(pixel, 24u);
            count = count + 1u;
        }
    }

    let half = count / 2u;
    let average_r = (sum_r + half) / count;
    let average_g = (sum_g + half) / count;
    let average_b = (sum_b + half) / count;
    let average_a = (sum_a + half) / count;
    let output_index = out_y * params.output_width + out_x;
    output[output_index] = pack_rgba(average_r, average_g, average_b, average_a);
}
"#;
