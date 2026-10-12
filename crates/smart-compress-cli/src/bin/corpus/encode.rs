//! Self-contained PNG and baseline JPEG writers for the synthetic corpus.
//!
//! The corpus hashes must not move when a codec dependency is upgraded, so the
//! bytes written here depend on this file alone. Everything is integer math.

const CRC_TABLE: [u32; 256] = {
    let mut table = [0_u32; 256];
    let mut index = 0;
    while index < 256 {
        let mut crc = index as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                0xEDB8_8320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[index] = crc;
        index += 1;
    }
    table
};

fn crc32(parts: &[&[u8]]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in parts.iter().copied().flatten() {
        crc = CRC_TABLE[((crc ^ u32::from(byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1_u32, 0_u32);
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += u32::from(byte);
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// LSB-first bit sink used by DEFLATE.
struct DeflateBits {
    out: Vec<u8>,
    acc: u64,
    bits: u32,
}

impl DeflateBits {
    fn put(&mut self, value: u32, count: u32) {
        self.acc |= u64::from(value) << self.bits;
        self.bits += count;
        while self.bits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.bits -= 8;
        }
    }

    /// Huffman codes are packed most-significant bit first.
    fn put_code(&mut self, code: u32, count: u32) {
        self.put(code.reverse_bits() >> (32 - count), count);
    }

    /// Fixed literal/length code from RFC 1951 section 3.2.6.
    fn put_symbol(&mut self, symbol: u32) {
        match symbol {
            0..=143 => self.put_code(0x30 + symbol, 8),
            144..=255 => self.put_code(0x190 + symbol - 144, 9),
            256..=279 => self.put_code(symbol - 256, 7),
            _ => self.put_code(0xC0 + symbol - 280, 8),
        }
    }
}

/// zlib stream with one fixed-Huffman DEFLATE block.
// Greedy single-probe LZ77 with fixed codes, so smooth photographic data
// barely shrinks. Dynamic Huffman would fix that if PNG photo sources matter.
pub fn zlib(data: &[u8]) -> Vec<u8> {
    let mut sink = DeflateBits {
        out: vec![0x78, 0x01],
        acc: 0,
        bits: 0,
    };
    sink.put(1, 1);
    sink.put(1, 2);
    let mut head = vec![u32::MAX; 1 << 15];
    let mut position = 0;
    while position < data.len() {
        let mut length = 0;
        let mut distance = 0;
        if position + 3 <= data.len() {
            let key = u32::from(data[position])
                | u32::from(data[position + 1]) << 8
                | u32::from(data[position + 2]) << 16;
            let slot = (key.wrapping_mul(0x9E37_79B1) >> 17) as usize;
            let previous = head[slot] as usize;
            head[slot] = position as u32;
            if previous < position && position - previous <= 32768 {
                let limit = (data.len() - position).min(258);
                length = (0..limit)
                    .take_while(|&offset| data[previous + offset] == data[position + offset])
                    .count();
                distance = position - previous;
            }
        }
        if length < 3 {
            sink.put_symbol(u32::from(data[position]));
            position += 1;
            continue;
        }
        let code = LENGTH_BASE
            .iter()
            .rposition(|&base| usize::from(base) <= length)
            .expect("match length is at least three");
        sink.put_symbol(257 + code as u32);
        sink.put(
            (length - usize::from(LENGTH_BASE[code])) as u32,
            u32::from(LENGTH_EXTRA[code]),
        );
        let code = DISTANCE_BASE
            .iter()
            .rposition(|&base| usize::from(base) <= distance)
            .expect("match distance is at least one");
        sink.put_code(code as u32, 5);
        sink.put(
            (distance - usize::from(DISTANCE_BASE[code])) as u32,
            u32::from(DISTANCE_EXTRA[code]),
        );
        position += length;
    }
    sink.put_symbol(256);
    if sink.bits > 0 {
        sink.out.push(sink.acc as u8);
    }
    sink.out.extend_from_slice(&adler32(data).to_be_bytes());
    sink.out
}

fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(&[kind, data]).to_be_bytes());
}

/// 8-bit RGB or RGBA PNG; `icc` is an optional (profile name, profile bytes).
// The filtered raster is held in memory (about 600 MB at 199.8 MP). Streaming
// rows into the compressor would avoid that if it ever matters.
pub fn png(
    width: u32,
    height: u32,
    alpha: bool,
    icc: Option<(&str, &[u8])>,
    pixel: impl Fn(u32, u32) -> [u8; 4],
) -> Vec<u8> {
    let channels = if alpha { 4 } else { 3 };
    let mut raw = Vec::with_capacity(height as usize * (1 + width as usize * channels));
    for y in 0..height {
        raw.push(1); // Sub filter
        let mut left = [0_u8; 4];
        for x in 0..width {
            let value = pixel(x, y);
            for (sample, previous) in value.iter().zip(left).take(channels) {
                raw.push(sample.wrapping_sub(previous));
            }
            left = value;
        }
    }
    let idat = zlib(&raw);
    assert!(
        idat.len() <= i32::MAX as usize,
        "IDAT exceeds one PNG chunk"
    );

    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, if alpha { 6 } else { 2 }, 0, 0, 0]);
    png_chunk(&mut out, b"IHDR", &header);
    if let Some((name, profile)) = icc {
        let mut data = name.as_bytes().to_vec();
        data.extend_from_slice(&[0, 0]);
        data.extend_from_slice(&zlib(profile));
        png_chunk(&mut out, b"iCCP", &data);
    }
    png_chunk(&mut out, b"IDAT", &idat);
    png_chunk(&mut out, b"IEND", &[]);
    out
}

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// ITU-T T.81 table K.1 scaled to libjpeg quality 90, in natural order.
const QUANT: [i64; 64] = {
    const BASE: [i64; 64] = [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ];
    let mut table = [0_i64; 64];
    let mut index = 0;
    while index < 64 {
        table[index] = (BASE[index] * 20 + 50) / 100;
        index += 1;
    }
    table
};

/// ITU-T T.81 tables K.3 and K.5 (luminance DC and AC), used for every component.
const DC_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
const AC_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

/// Canonical (code, length) per symbol for a JPEG Huffman table.
fn huffman_codes(bits: &[u8; 16], values: &[u8]) -> [(u32, u32); 256] {
    let mut codes = [(0, 0); 256];
    let mut code = 0;
    let mut next = values.iter();
    for (length, &count) in (1..).zip(bits) {
        for _ in 0..count {
            let symbol = *next.next().expect("Huffman table lists every symbol");
            codes[usize::from(symbol)] = (code, length);
            code += 1;
        }
        code <<= 1;
    }
    codes
}

/// MSB-first bit sink with JPEG 0xFF byte stuffing.
struct JpegBits {
    out: Vec<u8>,
    acc: u64,
    bits: u32,
}

impl JpegBits {
    fn put(&mut self, value: u32, count: u32) {
        self.acc = (self.acc << count) | u64::from(value);
        self.bits += count;
        while self.bits >= 8 {
            let byte = (self.acc >> (self.bits - 8)) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0);
            }
            self.bits -= 8;
        }
    }

    /// Huffman code for the magnitude category, then the magnitude bits.
    fn put_coefficient(&mut self, codes: &[(u32, u32); 256], run: u32, value: i64) {
        let size = 64 - value.unsigned_abs().leading_zeros();
        let (code, length) = codes[(run << 4 | size) as usize];
        self.put(code, length);
        let raw = if value < 0 { value - 1 } else { value };
        self.put((raw & ((1 << size) - 1)) as u32, size);
    }
}

/// `round(cos(k * pi / 16) * 2^14)` for k in 0..=8.
const COSINE: [i64; 9] = [16384, 16069, 15137, 13623, 11585, 9102, 6270, 3196, 0];

/// DCT-II basis scaled by 2^14, including the 1/sqrt(2) factor for u = 0.
fn dct_basis() -> [[i64; 8]; 8] {
    std::array::from_fn(|u| {
        std::array::from_fn(|x| {
            if u == 0 {
                return COSINE[4];
            }
            match (2 * x + 1) * u % 32 {
                m @ 0..=8 => COSINE[m],
                m @ 9..=16 => -COSINE[16 - m],
                m @ 17..=24 => -COSINE[m - 16],
                m => COSINE[32 - m],
            }
        })
    })
}

/// Forward DCT and quantization of one level-shifted block.
fn quantized_block(basis: &[[i64; 8]; 8], samples: &[i64; 64]) -> [i64; 64] {
    let mut rows = [0_i64; 64];
    for y in 0..8 {
        for u in 0..8 {
            rows[y * 8 + u] = (0..8).map(|x| samples[y * 8 + x] * basis[u][x]).sum();
        }
    }
    let mut out = [0_i64; 64];
    for v in 0..8 {
        for u in 0..8 {
            let sum: i64 = (0..8).map(|y| rows[y * 8 + u] * basis[v][y]).sum();
            // Two 2^14 basis scales plus the DCT's 1/4 normalization.
            let divisor = QUANT[v * 8 + u] << 30;
            out[v * 8 + u] = (sum + sum.signum() * (divisor / 2)) / divisor;
        }
    }
    out
}

fn jpeg_segment(out: &mut Vec<u8>, marker: u8, data: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&(data.len() as u16 + 2).to_be_bytes());
    out.extend_from_slice(data);
}

/// Adobe-style CMYK: every channel is stored inverted (255 means no ink).
fn rgb_to_inverted_cmyk([r, g, b, _]: [u8; 4]) -> [i64; 4] {
    let key = i64::from(r.max(g).max(b));
    let scale = |value: u8| {
        if key == 0 {
            255
        } else {
            (i64::from(value) * 255 + key / 2) / key
        }
    };
    [scale(r), scale(g), scale(b), key]
}

/// JFIF full-range BT.601.
fn rgb_to_ycbcr([r, g, b, _]: [u8; 4]) -> [i64; 4] {
    let (r, g, b) = (i64::from(r), i64::from(g), i64::from(b));
    [
        (19595 * r + 38470 * g + 7471 * b + 32768) >> 16,
        ((-11059 * r - 21709 * g + 32768 * b + 32767) >> 16) + 128,
        ((32768 * r - 27439 * g - 5329 * b + 32767) >> 16) + 128,
        0,
    ]
}

/// Baseline sequential 4:4:4 JPEG at quality 90. `cmyk` writes four components
/// with an Adobe APP14 segment (transform 0) instead of JFIF YCbCr.
pub fn jpeg(
    width: u32,
    height: u32,
    cmyk: bool,
    exif: Option<&[u8]>,
    pixel: impl Fn(u32, u32) -> [u8; 4],
) -> Vec<u8> {
    assert!(
        (1..=65535).contains(&width) && (1..=65535).contains(&height),
        "JPEG dimensions must be 1..=65535"
    );
    let components: u8 = if cmyk { 4 } else { 3 };
    let mut out = vec![0xFF, 0xD8];
    if cmyk {
        jpeg_segment(&mut out, 0xEE, b"Adobe\0\x64\0\0\0\0\0");
    } else {
        jpeg_segment(&mut out, 0xE0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
    }
    if let Some(exif) = exif {
        jpeg_segment(&mut out, 0xE1, &[&b"Exif\0\0"[..], exif].concat());
    }

    let mut table = vec![0];
    table.extend(ZIGZAG.iter().map(|&index| QUANT[index] as u8));
    jpeg_segment(&mut out, 0xDB, &table);

    let mut frame = vec![8];
    frame.extend_from_slice(&(height as u16).to_be_bytes());
    frame.extend_from_slice(&(width as u16).to_be_bytes());
    frame.push(components);
    for id in 1..=components {
        frame.extend_from_slice(&[id, 0x11, 0]);
    }
    jpeg_segment(&mut out, 0xC0, &frame);

    jpeg_segment(
        &mut out,
        0xC4,
        &[&[0x00][..], &DC_BITS, &DC_VALUES].concat(),
    );
    jpeg_segment(
        &mut out,
        0xC4,
        &[&[0x10][..], &AC_BITS, &AC_VALUES].concat(),
    );

    let mut scan = vec![components];
    for id in 1..=components {
        scan.extend_from_slice(&[id, 0x00]);
    }
    scan.extend_from_slice(&[0, 63, 0]);
    jpeg_segment(&mut out, 0xDA, &scan);

    let dc_codes = huffman_codes(&DC_BITS, &DC_VALUES);
    let ac_codes = huffman_codes(&AC_BITS, &AC_VALUES);
    let basis = dct_basis();
    let mut sink = JpegBits {
        out,
        acc: 0,
        bits: 0,
    };
    let mut predictions = [0_i64; 4];
    let mut blocks = [[0_i64; 64]; 4];
    for block_y in (0..height).step_by(8) {
        for block_x in (0..width).step_by(8) {
            for index in 0..64_u32 {
                // Edge blocks replicate the last row and column.
                let value = pixel(
                    (block_x + index % 8).min(width - 1),
                    (block_y + index / 8).min(height - 1),
                );
                let samples = if cmyk {
                    rgb_to_inverted_cmyk(value)
                } else {
                    rgb_to_ycbcr(value)
                };
                for (block, sample) in blocks.iter_mut().zip(samples) {
                    block[index as usize] = sample - 128;
                }
            }
            for (block, prediction) in blocks
                .iter()
                .zip(&mut predictions)
                .take(usize::from(components))
            {
                let quantized = quantized_block(&basis, block);
                sink.put_coefficient(&dc_codes, 0, quantized[0] - *prediction);
                *prediction = quantized[0];
                let mut run = 0;
                for &index in &ZIGZAG[1..] {
                    if quantized[index] == 0 {
                        run += 1;
                        continue;
                    }
                    for _ in 0..run / 16 {
                        sink.put(ac_codes[0xF0].0, ac_codes[0xF0].1);
                    }
                    sink.put_coefficient(&ac_codes, run % 16, quantized[index]);
                    run = 0;
                }
                if run > 0 {
                    sink.put(ac_codes[0x00].0, ac_codes[0x00].1);
                }
            }
        }
    }
    if sink.bits > 0 {
        let padding = 8 - sink.bits;
        sink.put((1 << padding) - 1, padding);
    }
    sink.out.extend_from_slice(&[0xFF, 0xD9]);
    sink.out
}

/// Minimal little-endian TIFF block holding only the EXIF orientation tag.
pub fn exif_orientation(orientation: u16) -> Vec<u8> {
    let mut exif = b"II*\0".to_vec();
    exif.extend_from_slice(&8_u32.to_le_bytes());
    exif.extend_from_slice(&1_u16.to_le_bytes());
    exif.extend_from_slice(&0x0112_u16.to_le_bytes());
    exif.extend_from_slice(&3_u16.to_le_bytes());
    exif.extend_from_slice(&1_u32.to_le_bytes());
    exif.extend_from_slice(&orientation.to_le_bytes());
    exif.extend_from_slice(&0_u16.to_le_bytes());
    exif.extend_from_slice(&0_u32.to_le_bytes());
    exif
}

/// D50-adapted colorant columns (red, green, blue) as s15Fixed16 XYZ.
pub const SRGB_COLORANTS: [[i32; 3]; 3] = [
    [0x6FA2, 0x38F5, 0x0390],
    [0x6299, 0xB785, 0x18DA],
    [0x24A0, 0x0F84, 0xB6CF],
];
pub const DISPLAY_P3_COLORANTS: [[i32; 3]; 3] = [
    [0x83DF, 0x3DBF, -0x45],
    [0x4ABF, 0xB137, 0x0AB9],
    [0x2838, 0x110B, 0xC8B9],
];

fn icc_xyz(values: [i32; 3]) -> Vec<u8> {
    let mut data = b"XYZ \0\0\0\0".to_vec();
    for value in values {
        data.extend_from_slice(&value.to_be_bytes());
    }
    data
}

/// ICC v2.1 display profile: three-component matrix/TRC with a gamma 2.2 curve
/// (an approximation of the piecewise sRGB transfer function).
pub fn icc_profile(description: &str, colorants: [[i32; 3]; 3]) -> Vec<u8> {
    let mut desc = b"desc\0\0\0\0".to_vec();
    desc.extend_from_slice(&(description.len() as u32 + 1).to_be_bytes());
    desc.extend_from_slice(description.as_bytes());
    // NUL terminator, empty Unicode and ScriptCode records.
    desc.extend_from_slice(&[0; 1 + 4 + 4 + 2 + 1 + 67]);
    let copyright = b"text\0\0\0\0Public domain synthetic test profile\0".to_vec();
    // u8Fixed8 gamma 2.2.
    let curve = b"curv\0\0\0\0\0\0\0\x01\x02\x33".to_vec();
    let [red, green, blue] = colorants;
    let tags: [(&[u8; 4], Vec<u8>); 9] = [
        (b"desc", desc),
        (b"cprt", copyright),
        (b"wtpt", icc_xyz([0xF351, 0x1_0000, 0x1_16CC])),
        (b"rXYZ", icc_xyz(red)),
        (b"gXYZ", icc_xyz(green)),
        (b"bXYZ", icc_xyz(blue)),
        (b"rTRC", curve.clone()),
        (b"gTRC", curve.clone()),
        (b"bTRC", curve),
    ];

    let mut directory = (tags.len() as u32).to_be_bytes().to_vec();
    let mut body = Vec::new();
    let body_start = 128 + 4 + 12 * tags.len();
    for (signature, data) in &tags {
        directory.extend_from_slice(*signature);
        directory.extend_from_slice(&((body_start + body.len()) as u32).to_be_bytes());
        directory.extend_from_slice(&(data.len() as u32).to_be_bytes());
        body.extend_from_slice(data);
        body.resize(body.len().next_multiple_of(4), 0);
    }

    let mut profile = vec![0_u8; 128];
    profile[0..4].copy_from_slice(&((body_start + body.len()) as u32).to_be_bytes());
    profile[8..12].copy_from_slice(&[2, 0x10, 0, 0]);
    profile[12..24].copy_from_slice(b"mntrRGB XYZ ");
    // Fixed creation date 2026-01-01 00:00:00.
    profile[24..30].copy_from_slice(&[0x07, 0xEA, 0, 1, 0, 1]);
    profile[36..40].copy_from_slice(b"acsp");
    // D50 PCS illuminant.
    profile[68..80].copy_from_slice(&[0, 0, 0xF6, 0xD6, 0, 1, 0, 0, 0, 0, 0xD3, 0x2D]);
    profile.extend_from_slice(&directory);
    profile.extend_from_slice(&body);
    profile
}
