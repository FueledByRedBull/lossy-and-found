/// Colour space an ICC profile describes (`RGB `, `GRAY`, `CMYK`, ...), or
/// `None` when the bytes are not an ICC profile at all.
pub fn icc_color_space(profile: &[u8]) -> Option<[u8; 4]> {
    (profile.len() >= 128 && &profile[36..40] == b"acsp")
        .then(|| [profile[16], profile[17], profile[18], profile[19]])
}

/// Whether a profile provably describes plain sRGB, so that a viewer which
/// finds no profile shows the same colours. Only matrix/TRC (RGB) and
/// single-curve (grey) profiles can be shown to be sRGB; anything else,
/// including every profile with lookup tables or that cannot be parsed, is
/// reported as not sRGB, which keeps it. The primaries must match sRGB to
/// 0.4% of XYZ and each curve the sRGB curve to 0.01 of linear light, so
/// gamma 2.2 stand-ins count as sRGB and gamma 1.8, 2.4, Display P3 and
/// Adobe RGB do not.
pub fn icc_is_plain_srgb(profile: &[u8]) -> bool {
    const SRGB_D50: [[f64; 3]; 3] = [
        [0.4361, 0.2225, 0.0139],
        [0.3851, 0.7169, 0.0971],
        [0.1431, 0.0606, 0.7141],
    ];
    let srgb_curve = |x: f64| {
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    };
    let curve_matches = |signature: &[u8; 4]| {
        icc_tag(profile, signature).is_some_and(|curve| {
            [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
                .iter()
                .all(|&x| icc_curve_at(curve, x).is_some_and(|y| (y - srgb_curve(x)).abs() <= 0.01))
        })
    };
    if icc_tag(profile, b"A2B0").is_some() {
        return false;
    }
    match icc_color_space(profile).as_ref() {
        Some(b"GRAY") => curve_matches(b"kTRC"),
        Some(b"RGB ") => {
            let colorant = |signature: &[u8; 4]| {
                let data = icc_tag(profile, signature)?;
                let value = |at: usize| {
                    Some(
                        f64::from(i32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?))
                            / 65536.0,
                    )
                };
                (data.starts_with(b"XYZ ")).then_some([value(8)?, value(12)?, value(16)?])
            };
            [b"rXYZ", b"gXYZ", b"bXYZ"]
                .iter()
                .zip(SRGB_D50)
                .all(|(signature, expected)| {
                    colorant(signature).is_some_and(|actual| {
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(a, e)| (a - e).abs() <= 0.004)
                    })
                })
                && [b"rTRC", b"gTRC", b"bTRC"].iter().all(|s| curve_matches(s))
        }
        _ => false,
    }
}

/// The data of one tag of an ICC profile.
fn icc_tag<'a>(profile: &'a [u8], signature: &[u8; 4]) -> Option<&'a [u8]> {
    let word = |at: usize| {
        Some(u32::from_be_bytes(
            profile.get(at..at + 4)?.try_into().ok()?,
        ))
    };
    (0..word(128)?.min(256) as usize).find_map(|index| {
        let entry = 132 + 12 * index;
        if profile.get(entry..entry + 4)? != signature {
            return None;
        }
        let offset = word(entry + 4)? as usize;
        profile.get(offset..offset.checked_add(word(entry + 8)? as usize)?)
    })
}

/// Value of a `curv` or `para` tag at `x` in 0..=1.
fn icc_curve_at(data: &[u8], x: f64) -> Option<f64> {
    let half = |at: usize| {
        Some(f64::from(u16::from_be_bytes(
            data.get(at..at + 2)?.try_into().ok()?,
        )))
    };
    let fixed = |at: usize| {
        Some(f64::from(i32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?)) / 65536.0)
    };
    match data.get(..4)? {
        b"curv" => {
            let count = u32::from_be_bytes(data.get(8..12)?.try_into().ok()?) as usize;
            match count {
                0 => Some(x),
                1 => Some(x.powf(half(12)? / 256.0)),
                _ => {
                    let position = x * (count - 1) as f64;
                    let low = position.floor() as usize;
                    let at = |index: usize| Some(half(12 + 2 * index)? / 65535.0);
                    let (a, b) = (at(low)?, at((low + 1).min(count - 1))?);
                    Some(a + (b - a) * (position - low as f64))
                }
            }
        }
        b"para" => {
            let function = u16::from_be_bytes(data.get(8..10)?.try_into().ok()?);
            let [g, a, b, c, d, e, f] =
                [0, 1, 2, 3, 4, 5, 6].map(|i| fixed(12 + 4 * i).unwrap_or(0.0));
            Some(match function {
                0 => x.powf(g),
                1 if x >= -b / a => (a * x + b).powf(g),
                1 => 0.0,
                2 if x >= -b / a => (a * x + b).powf(g) + c,
                2 => c,
                3 if x >= d => (a * x + b).powf(g),
                3 => c * x,
                4 if x >= d => (a * x + b).powf(g) + e,
                4 => c * x + f,
                _ => return None,
            })
        }
        _ => None,
    }
}
