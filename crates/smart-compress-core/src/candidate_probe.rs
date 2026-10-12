//! Header probes for encoded candidates.
//!
//! Dimensions, alpha and frame count are read in-process from the first
//! `PROBE_BYTES` of a candidate, so one that cannot match what was encoded
//! fails before a decoder process is started. The parsers take untrusted bytes:
//! every read is bounds-checked, nothing larger than the prefix is allocated,
//! and a file they cannot read gives `None`, which leaves the verdict to the
//! decoder.

use std::{fs::File, io::Read, path::Path};

use anyhow::{Result, ensure};
use image::{DynamicImage, GenericImageView};
use smart_compress_codecs::Codec;

/// Bytes read from the start of a candidate. The AVIF `meta` box and the leading
/// boxes of a JPEG XL container must end inside it. A file whose header boxes lie
/// further in (a very large ICC profile or Exif block ahead of the image data)
/// is not probed and goes straight to the decoder.
pub const PROBE_BYTES: u64 = 64 * 1024;

/// What a header says about the image. Fields the header does not state, or
/// that could not be read reliably, are `None`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CandidateHeader {
    /// Size of the decoded image. Left out when the file rotates or crops it.
    pub dimensions: Option<(u32, u32)>,
    /// An image sequence or animation rather than one still frame.
    pub animated: bool,
    pub alpha: Option<bool>,
    /// Bits per sample of the first colour channel.
    pub bit_depth: Option<u8>,
    /// Colour channels, without alpha (AVIF only).
    pub channels: Option<u8>,
    pub colour: Option<ColourInfo>,
}

/// Colour description boxes of an AVIF image (`colr`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColourInfo {
    pub icc: bool,
    pub nclx: bool,
}

/// Fails when the header of a candidate shows it cannot be the image that was
/// encoded. Anything that cannot be read passes.
pub fn check_candidate(
    codec: Codec,
    path: &Path,
    original: &DynamicImage,
    has_alpha: bool,
) -> Result<()> {
    let header = match codec {
        Codec::Avif => read_prefix(path).and_then(|data| probe_avif(&data)),
        Codec::Jxl => read_prefix(path).and_then(|data| probe_jxl(&data)),
        // The image crate reads the header of these without decoding.
        Codec::Png | Codec::Jpeg | Codec::Webp => image::ImageReader::open(path)
            .and_then(image::ImageReader::with_guessed_format)
            .ok()
            .and_then(|reader| reader.into_dimensions().ok())
            .map(|dimensions| CandidateHeader {
                dimensions: Some(dimensions),
                ..CandidateHeader::default()
            }),
    };
    let Some(header) = header else {
        return Ok(());
    };
    ensure!(
        !header.animated,
        "candidate header declares an image sequence or animation, but one frame was encoded; decoder not run"
    );
    if let Some((width, height)) = header.dimensions {
        let (expected_width, expected_height) = original.dimensions();
        ensure!(
            (width, height) == (expected_width, expected_height),
            "candidate header says {width}x{height} but {expected_width}x{expected_height} was encoded; decoder not run"
        );
    }
    // Encoders may drop the alpha channel of a fully opaque image; only a lost
    // channel that carried transparency is a mismatch.
    ensure!(
        !(has_alpha
            && codec.supports_alpha()
            && header.alpha == Some(false)
            && has_transparency(original)),
        "candidate header has no alpha channel but the source has transparency; decoder not run"
    );
    Ok(())
}

fn has_transparency(image: &DynamicImage) -> bool {
    image.color().has_alpha() && image.to_rgba16().pixels().any(|pixel| pixel[3] != u16::MAX)
}

/// The first `PROBE_BYTES` of a file.
pub fn read_prefix(path: &Path) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)
        .ok()?
        .take(PROBE_BYTES)
        .read_to_end(&mut bytes)
        .ok()?;
    Some(bytes)
}

/// Reads an AVIF file's `ftyp` and `meta` boxes. `None` when the data does not
/// start with an AVIF `ftyp`. When `meta` is missing or cut off, only the
/// still-versus-sequence answer, which the `ftyp` brands give, is filled in.
pub fn probe_avif(data: &[u8]) -> Option<CandidateHeader> {
    let mut top = raw_boxes(data);
    let (kind, ftyp, _) = top.next()?;
    if kind != b"ftyp" {
        return None;
    }
    // Major brand, four bytes of minor version, then the compatible brands.
    let brands = || {
        ftyp.as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .filter_map(|(index, brand)| (index != 1).then_some(brand))
    };
    let animated = brands().any(|brand| brand == b"avis");
    if !animated && !brands().any(|brand| brand == b"avif") {
        return None;
    }
    let mut header = CandidateHeader {
        animated,
        ..CandidateHeader::default()
    };
    if let Some((_, meta, true)) = top.find(|(kind, ..)| *kind == b"meta") {
        let _ = read_avif_meta(meta, &mut header);
    }
    Some(header)
}

/// Fills `header` from an AVIF `meta` box. Stops at the first thing it cannot
/// read; what was filled in before that stays.
fn read_avif_meta(meta: &[u8], header: &mut CandidateHeader) -> Option<()> {
    let (mut primary, mut ipco, mut ipma) = (None, None, None);
    // `meta` is a full box: version and flags come before its children.
    for (kind, body) in boxes(meta.get(4..)?) {
        match kind {
            b"pitm" => {
                let mut body = Cursor(body);
                let version = body.uint(1)?;
                body.take(3)?;
                primary = body.uint(if version == 0 { 2 } else { 4 });
            }
            b"iprp" => {
                for (kind, body) in boxes(body) {
                    match kind {
                        b"ipco" => ipco = Some(body),
                        b"ipma" => ipma = Some(body),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    let properties = boxes(ipco?).collect::<Vec<_>>();
    // The alpha plane is an auxiliary image that names itself in an `auxC` property.
    header.alpha = Some(properties.iter().any(|&(kind, body)| {
        kind == b"auxC"
            && body.get(4..).is_some_and(|urn| {
                urn.starts_with(b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha")
                    || urn.starts_with(b"urn:mpeg:hevc:2015:auxid:1")
            })
    }));

    // Properties of the primary image, from its `ipma` association list.
    let mut ipma = Cursor(ipma?);
    let version = ipma.uint(1)?;
    let wide = ipma.uint(3)? & 1 == 1;
    let mut linked = Vec::new();
    for _ in 0..ipma.uint(4)? {
        let item = ipma.uint(if version == 0 { 2 } else { 4 })?;
        let count = ipma.uint(1)?;
        let mut indices = Vec::new();
        for _ in 0..count {
            // An essential flag, then a 1-based property index.
            let entry = ipma.uint(if wide { 2 } else { 1 })?;
            indices.push((entry & if wide { 0x7fff } else { 0x7f }) as usize);
        }
        if primary.is_none_or(|primary| primary == item) {
            linked = indices;
            break;
        }
    }
    let primary_properties = linked
        .iter()
        .filter_map(|&index| properties.get(index.checked_sub(1)?).copied())
        .collect::<Vec<_>>();
    if primary_properties.is_empty() {
        return None;
    }
    let property = |name: &[u8; 4]| {
        primary_properties
            .iter()
            .find(|property| property.0 == name)
            .map(|property| property.1)
    };

    let mut colour = ColourInfo {
        icc: false,
        nclx: false,
    };
    for (_, body) in primary_properties
        .iter()
        .filter(|(kind, _)| *kind == b"colr")
    {
        colour.icc |= body.starts_with(b"rICC") || body.starts_with(b"prof");
        colour.nclx |= body.starts_with(b"nclx");
    }
    header.colour = Some(colour);
    if let Some(body) = property(b"pixi") {
        let mut body = Cursor(body);
        body.take(4)?;
        header.channels = u8::try_from(body.uint(1)?).ok();
        header.bit_depth = u8::try_from(body.uint(1)?).ok();
    }
    // Rotation and cropping are properties, not pixels; whether the decoder
    // applies them is not worth guessing, so the size is left unknown.
    if property(b"irot").is_none() && property(b"imir").is_none() && property(b"clap").is_none() {
        let mut ispe = Cursor(property(b"ispe")?);
        ispe.take(4)?;
        header.dimensions = Some((ispe.uint(4)?, ispe.uint(4)?));
    }
    Some(())
}

/// Reads the image header of a JPEG XL file, bare or in its box container.
/// Dimensions come from the size header. Animation, bit depth and alpha come
/// from the image metadata that follows it; they stay unset when it uses a
/// feature this reader does not follow (a preview, or a colour-filter-array
/// channel ahead of any alpha).
pub fn probe_jxl(data: &[u8]) -> Option<CandidateHeader> {
    let mut stream = Vec::new();
    if data.starts_with(&[0xFF, 0x0A]) {
        stream.extend_from_slice(data);
    } else {
        let mut container = raw_boxes(data);
        container.next().filter(|(kind, ..)| *kind == b"JXL ")?;
        // The codestream is one `jxlc` box or a run of `jxlp` boxes, each
        // behind a four-byte index.
        for (kind, body, _) in container {
            match kind {
                b"jxlc" => {
                    stream.extend_from_slice(body);
                    break;
                }
                b"jxlp" => stream.extend_from_slice(body.get(4..)?),
                _ => {}
            }
        }
    }
    let mut bits = Bits {
        data: stream.strip_prefix(&[0xFF, 0x0A])?,
        position: 0,
    };
    let mut header = CandidateHeader {
        dimensions: Some(bits.size_header()?),
        ..CandidateHeader::default()
    };
    let _ = read_jxl_metadata(&mut bits, &mut header);
    Some(header)
}

/// Fills `header` from the JPEG XL `ImageMetadata` bundle, in order, stopping
/// at the first thing it cannot read.
fn read_jxl_metadata(bits: &mut Bits, header: &mut CandidateHeader) -> Option<()> {
    if bits.flag()? {
        // Every field at its default: upright, still, 8-bit, no extra channels.
        header.alpha = Some(false);
        header.bit_depth = Some(8);
        return Some(());
    }
    if bits.flag()? {
        // The decoder applies the orientation, which may swap the sides.
        if bits.read(3)? != 0 {
            header.dimensions = None;
        }
        if bits.flag()? {
            bits.size_header()?;
        }
        if bits.flag()? {
            return None;
        }
        if bits.flag()? {
            header.animated = true;
            return Some(());
        }
    }
    let depth = bits.bit_depth()?;
    header.bit_depth = u8::try_from(depth).ok();
    bits.flag()?;
    let channels = bits.u32([(0, 0), (0, 1), (4, 2), (12, 1)])?;
    for _ in 0..channels {
        // A channel that leaves everything at its default is alpha.
        if bits.flag()? {
            header.alpha = Some(true);
            return Some(());
        }
        let kind = bits.u32([(0, 0), (0, 1), (4, 2), (6, 18)])?;
        if kind == 0 {
            header.alpha = Some(true);
            return Some(());
        }
        // Not alpha: step over the rest of this channel's description.
        bits.bit_depth()?;
        bits.u32([(0, 0), (0, 3), (0, 4), (3, 1)])?;
        let name_length = bits.u32([(0, 0), (4, 0), (5, 16), (10, 48)])?;
        bits.skip(8 * u64::from(name_length))?;
        match kind {
            // Spot colour: four half-precision floats.
            3 => bits.skip(64)?,
            // Colour-filter-array index.
            5 => return None,
            _ => {}
        }
    }
    header.alpha = Some(false);
    Some(())
}

/// Reads bits least significant first, the order JPEG XL headers use.
struct Bits<'a> {
    data: &'a [u8],
    position: usize,
}

impl Bits<'_> {
    fn read(&mut self, count: u32) -> Option<u64> {
        let mut value = 0;
        for index in 0..count {
            let byte = *self.data.get(self.position / 8)?;
            value |= u64::from((byte >> (self.position % 8)) & 1) << index;
            self.position += 1;
        }
        Some(value)
    }

    fn flag(&mut self) -> Option<bool> {
        Some(self.read(1)? == 1)
    }

    fn skip(&mut self, count: u64) -> Option<()> {
        let end = usize::try_from(count).ok()?.checked_add(self.position)?;
        (end <= self.data.len() * 8).then(|| self.position = end)
    }

    /// A `U32` field: a two-bit choice of (extra bits, offset) pairs.
    fn u32(&mut self, choices: [(u32, u32); 4]) -> Option<u32> {
        let (count, offset) = choices[self.read(2)? as usize];
        u32::try_from(u64::from(offset) + self.read(count)?).ok()
    }

    /// A `BitDepth` bundle; returns the bits per sample.
    fn bit_depth(&mut self) -> Option<u32> {
        if self.flag()? {
            let depth = self.u32([(0, 32), (0, 16), (0, 24), (6, 1)])?;
            self.read(4)?;
            Some(depth)
        } else {
            self.u32([(0, 8), (0, 10), (0, 12), (6, 1)])
        }
    }

    /// A `SizeHeader`: height, then width unless an aspect ratio gives it.
    fn size_header(&mut self) -> Option<(u32, u32)> {
        let small = self.flag()?;
        let side = |bits: &mut Self| {
            if small {
                Some((bits.read(5)? as u32 + 1) * 8)
            } else {
                bits.u32([(9, 1), (13, 1), (18, 1), (30, 1)])
            }
        };
        let height = side(self)?;
        let width = match self.read(3)? {
            0 => side(self)?,
            ratio => {
                let (numerator, denominator) =
                    [(1, 1), (12, 10), (4, 3), (3, 2), (16, 9), (5, 4), (2, 1)][ratio as usize - 1];
                u32::try_from(u64::from(height) * numerator / denominator).ok()?
            }
        };
        Some((width, height))
    }
}

struct Cursor<'a>(&'a [u8]);

impl<'a> Cursor<'a> {
    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(count)?;
        self.0 = rest;
        Some(head)
    }

    /// A big-endian integer of up to four bytes.
    fn uint(&mut self, count: usize) -> Option<u32> {
        Some(
            self.take(count)?
                .iter()
                .fold(0, |value, byte| (value << 8) | u32::from(*byte)),
        )
    }
}

/// The ISO base media boxes at one level of `data`, each as (type, body,
/// complete). A box that runs past the end of the data comes last, with the
/// part that is there and `complete` false. Reading stops at a malformed header.
fn raw_boxes(mut rest: &[u8]) -> impl Iterator<Item = (&[u8], &[u8], bool)> {
    std::iter::from_fn(move || {
        let size = u32::from_be_bytes(rest.get(..4)?.try_into().ok()?);
        let kind = rest.get(4..8)?;
        let (header, size) = match size {
            // The box runs to the end of the data.
            0 => (8, rest.len() as u64),
            1 => (16, u64::from_be_bytes(rest.get(8..16)?.try_into().ok()?)),
            size => (8, u64::from(size)),
        };
        let size = usize::try_from(size).ok().filter(|size| *size >= header)?;
        let end = size.min(rest.len());
        let body = rest.get(header..end)?;
        rest = &rest[end..];
        Some((kind, body, size <= end))
    })
}

/// Like `raw_boxes`, without the box that is cut off.
fn boxes(data: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> {
    raw_boxes(data).map_while(|(kind, body, complete)| complete.then_some((kind, body)))
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage, Rgba, RgbaImage};

    use super::*;

    fn hex(text: &str) -> Vec<u8> {
        let digits = text.split_whitespace().collect::<String>();
        (0..digits.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&digits[index..index + 2], 16).unwrap())
            .collect()
    }

    // Captured with tools/libavif/avifenc.exe (-q 50 --speed 10) from a 37x21
    // gradient, cut after the `meta` box.
    fn avif_rgb() -> Vec<u8> {
        hex(
            "00000020667479706176696600000000617669666d6966316d6961664d413141
             000000eb6d657461000000000000002168646c72000000000000000070696374
             000000000000000000000000000000000e7069746d0000000000010000001e69
             6c6f63000000004400000100010000000100000113000000830000002869696e
             660000000000010000001a696e6665020000000001000061763031436f6c6f72
             000000006a697072700000004b6970636f000000146973706500000000000000
             2500000015000000107069786900000000030808080000000c61763143812000
             0000000013636f6c726e636c780001000d0006800000001769706d6100000000
             0000000100010401028304",
        )
    }

    // The same gradient with a varying alpha channel.
    fn avif_rgba() -> Vec<u8> {
        hex(
            "00000020667479706176696600000000617669666d6966316d6961664d413141
             000001866d657461000000000000002168646c72000000000000000070696374
             000000000000000000000000000000000e7069746d0000000000010000002c69
             6c6f6300000000440000020001000000010000024e0000008400020000000100
             0001ae000000a00000004269696e660000000000020000001a696e6665020000
             000001000061763031436f6c6f72000000001a696e6665020000000002000061
             763031416c706861000000001a69726566000000000000000e6175786c000200
             010001000000c3697072700000009d6970636f00000014697370650000000000
             00002500000015000000107069786900000000030808080000000c6176314381
             20000000000013636f6c726e636c780001000d0006800000000e706978690000
             000001080000000c6176314381001c0000000038617578430000000075726e3a
             6d7065673a6d706567423a636963703a73797374656d733a617578696c696172
             793a616c706861000000001e69706d6100000000000000020001040102830400
             020401058607",
        )
    }

    // Captured with tools/libjxl/bin/cjxl.exe (-q 60 -e 3), cut after the header.
    fn jxl_rgb() -> Vec<u8> {
        hex(
            "ff0aa0d0010014080200a4026b570a00800a9551c60d5ecef57cf9a1c362dac6
             7586b6dab6b02451",
        )
    }

    fn jxl_rgba() -> Vec<u8> {
        hex(
            "ff0aa050b013005080200010076b570a00800a9551c60d5ecef57cf9a1c362da
             c67586b6dab6b024910099b7720178f51e0283c1a024a4a0a0c006cbf531726e",
        )
    }

    fn jxl_container() -> Vec<u8> {
        hex(
            "0000000c4a584c200d0a870a00000014667479706a786c20000000006a786c20
             000000bd6a786c63ff0aa0d0010014080200a4026b570a00800a9551c60d5ece
             f57cf9a1c362dac67586b6dab6b02451641c9ba41338cc24481c",
        )
    }

    // A JPEG with Exif orientation 6 converted by cjxl: the codestream is split
    // over two `jxlp` boxes with an Exif box between them, and carries the
    // orientation.
    fn jxl_oriented() -> Vec<u8> {
        hex(
            "0000000c4a584c200d0a870a00000014667479706a786c20000000006a786c20
             000000156a786c7000000000ff0af83b0b241497040000002e62726f62457869
             661b1d0000049c631341cc2ff1a9091b702201421ae022778034a9f0dbd317c0
             4e4e0800000b5c6a786c70800000010014080200d8d2e5c0",
        )
    }

    #[test]
    fn still_avif_without_alpha() {
        let header = probe_avif(&avif_rgb()).unwrap();
        assert_eq!(
            header,
            CandidateHeader {
                dimensions: Some((37, 21)),
                animated: false,
                alpha: Some(false),
                bit_depth: Some(8),
                channels: Some(3),
                colour: Some(ColourInfo {
                    icc: false,
                    nclx: true
                }),
            }
        );
    }

    #[test]
    fn still_avif_with_alpha() {
        let header = probe_avif(&avif_rgba()).unwrap();
        assert_eq!(header.dimensions, Some((37, 21)));
        assert_eq!(header.alpha, Some(true));
        assert_eq!(header.channels, Some(3));
        assert!(!header.animated);
    }

    #[test]
    fn avif_sequences_are_detected_by_brand() {
        let still = b"\0\0\0\x1cftypavif\0\0\0\0avifmif1miaf";
        let sequence = b"\0\0\0\x20ftypavis\0\0\0\0avifavismsf1miaf";
        let compatible_only = b"\0\0\0\x1cftypavif\0\0\0\0mif1avismiaf";
        assert!(!probe_avif(still).unwrap().animated);
        assert!(probe_avif(sequence).unwrap().animated);
        assert!(probe_avif(compatible_only).unwrap().animated);
        assert!(probe_avif(b"\x89PNG").is_none());
        // A brand cannot be the minor version.
        assert!(probe_avif(b"\0\0\0\x14ftypmif1avis\0\0\0\0").is_none());
        // A sequence is reported even when its `meta` box is missing.
        assert_eq!(
            probe_avif(sequence).unwrap().dimensions,
            None,
            "no meta box, no size"
        );
    }

    #[test]
    fn avif_with_a_cut_off_meta_box_reports_only_its_brands() {
        let data = avif_rgb();
        // The `ftyp` box is the first 32 bytes.
        for length in 32..data.len() {
            let header = probe_avif(&data[..length]).unwrap();
            assert!(!header.animated);
            assert_eq!(header.dimensions, None, "length {length}");
            assert_eq!(header.alpha, None);
        }
    }

    fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    fn ispe(width: u32, height: u32) -> (&'static [u8; 4], Vec<u8>) {
        let body = [
            vec![0; 4],
            width.to_be_bytes().into(),
            height.to_be_bytes().into(),
        ];
        (b"ispe", body.concat())
    }

    /// An AVIF with the given `ipco` properties, of which the primary item
    /// (number 1) is associated with the 1-based indices in `linked`.
    fn synthetic_avif(properties: &[(&[u8; 4], Vec<u8>)], linked: &[u16], wide: bool) -> Vec<u8> {
        let ipco = properties
            .iter()
            .flat_map(|(kind, body)| bx(kind, body))
            .collect::<Vec<_>>();
        let mut ipma = vec![0, 0, 0, u8::from(wide)];
        ipma.extend_from_slice(&1_u32.to_be_bytes());
        ipma.extend_from_slice(&1_u16.to_be_bytes());
        ipma.push(linked.len() as u8);
        for index in linked {
            if wide {
                ipma.extend_from_slice(&(0x8000 | index).to_be_bytes());
            } else {
                ipma.push(0x80 | *index as u8);
            }
        }
        let iprp = [bx(b"ipco", &ipco), bx(b"ipma", &ipma)].concat();
        let meta = [
            vec![0; 4],
            bx(b"pitm", &[0, 0, 0, 0, 0, 1]),
            bx(b"iprp", &iprp),
        ]
        .concat();
        [bx(b"ftyp", b"avif\0\0\0\0avifmif1"), bx(b"meta", &meta)].concat()
    }

    #[test]
    fn avif_properties_come_from_the_primary_item() {
        let alpha_urn = [
            vec![0; 4],
            b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0".to_vec(),
        ]
        .concat();
        let depth_urn = [
            vec![0; 4],
            b"urn:mpeg:mpegB:cicp:systems:auxiliary:depth\0".to_vec(),
        ]
        .concat();
        let read = |properties: &[(&[u8; 4], Vec<u8>)], linked: &[u16], wide| {
            probe_avif(&synthetic_avif(properties, linked, wide)).unwrap()
        };

        // The size of the primary item, not the first `ispe`; both index widths.
        let sizes = [ispe(10, 10), ispe(37, 21)];
        assert_eq!(read(&sizes, &[2], false).dimensions, Some((37, 21)));
        assert_eq!(read(&sizes, &[2], true).dimensions, Some((37, 21)));
        assert_eq!(read(&sizes, &[1], false).dimensions, Some((10, 10)));

        // Rotation, mirroring and cropping leave the size unknown.
        for name in [b"irot", b"imir", b"clap"] {
            let header = read(&[ispe(37, 21), (name, vec![0; 4])], &[1, 2], false);
            assert_eq!(header.dimensions, None, "{name:?}");
            assert_eq!(header.alpha, Some(false));
        }

        // Alpha is an auxiliary image named by its `auxC`; depth is not alpha.
        let with_alpha = [ispe(8, 8), (b"auxC", alpha_urn)];
        assert_eq!(read(&with_alpha, &[1], false).alpha, Some(true));
        let with_depth = [ispe(8, 8), (b"auxC", depth_urn)];
        assert_eq!(read(&with_depth, &[1], false).alpha, Some(false));

        // Channel count, bit depth and colour description.
        let described = [
            ispe(8, 8),
            (b"pixi", vec![0, 0, 0, 0, 1, 10]),
            (b"colr", b"prof".to_vec()),
        ];
        let header = read(&described, &[1, 2, 3], false);
        assert_eq!((header.channels, header.bit_depth), (Some(1), Some(10)));
        assert_eq!(
            header.colour,
            Some(ColourInfo {
                icc: true,
                nclx: false
            })
        );

        // No properties for the primary item: nothing is claimed.
        let header = read(&[ispe(8, 8)], &[], false);
        assert_eq!((header.dimensions, header.colour), (None, None));
    }

    #[test]
    fn bare_jxl_codestreams() {
        let header = probe_jxl(&jxl_rgb()).unwrap();
        assert_eq!(header.dimensions, Some((37, 21)));
        assert_eq!(header.alpha, Some(false));
        assert_eq!(header.bit_depth, Some(8));
        assert!(!header.animated);

        let header = probe_jxl(&jxl_rgba()).unwrap();
        assert_eq!(header.dimensions, Some((37, 21)));
        assert_eq!(header.alpha, Some(true));

        // 30x20 grey (small header), 20x10 at 16 bits, 24x16 animation,
        // 48x27 and 64x48 from aspect ratios.
        let cases = [
            (
                "ff0a9840901437020014080200a4046b570a00800a9551c60d5ecef57cf9a1c3",
                (30, 20),
                Some(8),
                false,
            ),
            (
                "ff0a4870fc64020014080200c0026b570a00800a9551c60d5ecef57cf9a1c362",
                (20, 10),
                Some(16),
                false,
            ),
            (
                "ff0ad0d0010014080200fc046b570a00800a9551c60d5ecef57cf9a1c362dac6",
                (48, 27),
                Some(8),
                false,
            ),
            (
                "ff0acb060014080200e00b6b570a00800a9551c60d5ecef57cf9a1c362dac675",
                (64, 48),
                Some(8),
                false,
            ),
            (
                "ff0a030503c8090014904c400098016b570a00800a9551c60d5ecef57cf9a1c3",
                (24, 16),
                None,
                true,
            ),
        ];
        for (data, dimensions, bit_depth, animated) in cases {
            let header = probe_jxl(&hex(data)).unwrap();
            assert_eq!(header.dimensions, Some(dimensions), "{data}");
            assert_eq!(header.bit_depth, bit_depth, "{data}");
            assert_eq!(header.animated, animated, "{data}");
        }
    }

    #[test]
    fn containerised_jxl_and_split_codestreams() {
        let header = probe_jxl(&jxl_container()).unwrap();
        assert_eq!(header.dimensions, Some((37, 21)));
        assert_eq!(header.alpha, Some(false));

        // The orientation swaps the sides on decode, so no size is claimed;
        // the header is split over two `jxlp` boxes.
        let header = probe_jxl(&jxl_oriented()).unwrap();
        assert_eq!(header.dimensions, None);
        assert!(!header.animated);
    }

    #[test]
    fn an_extra_alpha_channel_is_read_from_the_metadata() {
        // A 32x24 image with metadata that is not all default and one extra
        // channel that is: assembled field by field.
        let mut bits = Vec::new();
        let push = |bits: &mut Vec<bool>, value: u64, count: u32| {
            for index in 0..count {
                bits.push((value >> index) & 1 == 1);
            }
        };
        push(&mut bits, 1, 1); // small
        push(&mut bits, 2, 5); // height 24
        push(&mut bits, 0, 3); // explicit width
        push(&mut bits, 3, 5); // width 32
        push(&mut bits, 0, 1); // not all default
        push(&mut bits, 0, 1); // no extra fields
        push(&mut bits, 0, 1); // integer samples
        push(&mut bits, 0, 2); // 8 bits
        push(&mut bits, 1, 1); // 16-bit buffer sufficient
        push(&mut bits, 1, 2); // one extra channel
        push(&mut bits, 1, 1); // ... with every field default
        let mut data = vec![0xFF, 0x0A];
        data.extend(bits.chunks(8).map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0_u8, |byte, (index, bit)| byte | (u8::from(*bit) << index))
        }));
        let header = probe_jxl(&data).unwrap();
        assert_eq!(header.dimensions, Some((32, 24)));
        assert_eq!(header.alpha, Some(true));
        assert_eq!(header.bit_depth, Some(8));
    }

    #[test]
    fn truncated_garbage_and_hostile_input_never_panic() {
        assert!(probe_jxl(&[]).is_none());
        assert!(probe_jxl(&[0xFF, 0x0A]).is_none());
        assert!(probe_jxl(&jxl_rgb()[..3]).is_none());
        assert!(probe_avif(&[]).is_none());
        assert!(probe_avif(&[0; 7]).is_none());
        assert!(probe_avif(b"not an image at all, just some text").is_none());
        assert!(probe_jxl(b"not an image at all, just some text").is_none());
        // Oversized and undersized box lengths.
        assert!(probe_avif(&hex("ffffffff66747970 61766966 00000000")).is_some());
        assert!(probe_avif(&hex("00000004667479706176696600000000")).is_none());
        assert!(probe_avif(&hex("00000001667479700000000000000000")).is_none());

        // Every cut and every flipped byte of the real files.
        let files = [
            avif_rgb(),
            avif_rgba(),
            jxl_rgb(),
            jxl_rgba(),
            jxl_container(),
            jxl_oriented(),
        ];
        for data in &files {
            for length in 0..data.len() {
                probe_avif(&data[..length]);
                probe_jxl(&data[..length]);
            }
            for index in 0..data.len() {
                for flip in [0x01, 0x80, 0xFF] {
                    let mut copy = data.clone();
                    copy[index] ^= flip;
                    probe_avif(&copy);
                    probe_jxl(&copy);
                }
            }
        }
    }

    #[test]
    fn candidate_headers_are_checked_against_the_source() {
        let dir = tempfile::tempdir().unwrap();
        let opaque = DynamicImage::ImageRgb8(RgbImage::from_pixel(37, 21, Rgb([1, 2, 3])));
        let clear = DynamicImage::ImageRgba8(RgbaImage::from_pixel(37, 21, Rgba([1, 2, 3, 0])));
        let solid = DynamicImage::ImageRgba8(RgbaImage::from_pixel(37, 21, Rgba([1, 2, 3, 255])));
        let write = |name: &str, data: &[u8]| {
            let path = dir.path().join(name);
            std::fs::write(&path, data).unwrap();
            path
        };

        let rgb = write("rgb.jxl", &jxl_rgb());
        check_candidate(Codec::Jxl, &rgb, &opaque, false).unwrap();
        let wrong = DynamicImage::ImageRgb8(RgbImage::new(21, 37));
        let error = check_candidate(Codec::Jxl, &rgb, &wrong, false).unwrap_err();
        assert!(error.to_string().contains("37x21"), "{error}");

        // A lost alpha channel only matters when it carried transparency.
        assert!(check_candidate(Codec::Jxl, &rgb, &clear, true).is_err());
        check_candidate(Codec::Jxl, &rgb, &solid, true).unwrap();
        let rgba = write("rgba.jxl", &jxl_rgba());
        check_candidate(Codec::Jxl, &rgba, &clear, true).unwrap();
        let avif = write("rgb.avif", &avif_rgb());
        assert!(check_candidate(Codec::Avif, &avif, &clear, true).is_err());
        check_candidate(Codec::Avif, &avif, &opaque, false).unwrap();

        let animated = write(
            "sequence.avif",
            b"\0\0\0\x20ftypavis\0\0\0\0avifavismsf1miaf",
        );
        let error = check_candidate(Codec::Avif, &animated, &opaque, false).unwrap_err();
        assert!(error.to_string().contains("sequence"), "{error}");

        // Formats the image crate reads: header size, and unreadable files pass.
        let png = dir.path().join("candidate.png");
        opaque.save(&png).unwrap();
        check_candidate(Codec::Png, &png, &opaque, false).unwrap();
        assert!(check_candidate(Codec::Png, &png, &wrong, false).is_err());
        let junk = write("junk.jpg", b"definitely not a jpeg");
        check_candidate(Codec::Jpeg, &junk, &opaque, false).unwrap();
        let missing = dir.path().join("missing.webp");
        check_candidate(Codec::Webp, &missing, &opaque, false).unwrap();
        check_candidate(Codec::Avif, &missing, &opaque, false).unwrap();
    }
}
