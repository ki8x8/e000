use anyhow::{anyhow, Result};
use bincode::{deserialize, serialize};
use clap::{Parser, Subcommand};
use fs2::FileExt;
use hex::encode;
use rand::rngs::OsRng;
use rand::RngCore;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

// ---------- Constants ----------

/// Magic bytes identifying the TF512v1 container format.
const MAGIC: [u8; 8] = *b"TF512v1\0";
/// Length of the salt in bytes.
const SALT_LEN: usize = 16;
/// Length of the nonce in bytes.
const NONCE_LEN: usize = 16;
/// Length of the MAC tag in bytes.
const TAG_LEN: usize = 32;
/// Chunk size for streaming I/O (1 MiB).
const CHUNK: usize = 1 << 20;
/// Maximum attempts to generate a unique nonce.
const MAX_NONCE_ATTEMPTS: usize = 10;

// ---------- Nonce Management ----------

/// Manages a store of used nonces to prevent reuse.
struct NonceStore {
    nonces: Vec<[u8; NONCE_LEN]>,
    path: PathBuf,
}

impl NonceStore {
    /// Loads or creates a nonce store at `key_path` with `.tf512` extension.
    fn new(key_path: &Path) -> Result<Self> {
        let dir = key_path.parent().unwrap_or(Path::new("."));
        let nonce_path = dir.join("nonces.tf512");
        let nonces = if nonce_path.exists() {
            let file = File::open(&nonce_path)?;
            file.lock_exclusive()?;
            let data = fs::read(&nonce_path)?;
            deserialize(&data).map_err(|e| anyhow!("Failed to read nonce store: {}", e))?
        } else {
            Vec::new()
        };
        Ok(Self {
            nonces,
            path: nonce_path,
        })
    }

    /// Generates a unique nonce, checking against used nonces.
    fn generate_unique_nonce(&mut self) -> Result<[u8; NONCE_LEN]> {
        for _ in 0..MAX_NONCE_ATTEMPTS {
            let mut nonce = [0u8; NONCE_LEN];
            OsRng.fill_bytes(&mut nonce);
            if !self.nonces.iter().any(|n| bool::from(n.ct_eq(&nonce))) {
                self.nonces.push(nonce);
                return Ok(nonce);
            }
        }
        Err(anyhow!(
            "Failed to generate unique nonce after {} attempts. Consider creating a new key file.",
            MAX_NONCE_ATTEMPTS
        ))
    }

    /// Saves the nonce store atomically.
    fn save(&self) -> Result<()> {
        let tmp = temp_path_near(&self.path);
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp)?;
        file.lock_exclusive()?;
        let data = serialize(&self.nonces)?;
        file.write_all(&data)?;
        file.sync_all()?;
        drop(file);
        atomic_replace(&tmp, &self.path)?;
        Ok(())
    }
}

// ---------- Threefish-512 core (72 rounds, Skein v1.3 constants) ----------

/// Constant for key schedule parity computation.
const C240: u64 = 0x1BD11BDAA9FC1A22;
/// Rotation constants for Threefish-512 MIX operations.
const R512: [[u32; 4]; 8] = [
    [46, 36, 19, 37],
    [33, 27, 14, 42],
    [17, 49, 36, 39],
    [44, 9, 54, 56],
    [39, 30, 34, 24],
    [13, 50, 10, 17],
    [25, 29, 39, 43],
    [8, 35, 56, 22],
];
/// Permutation indices for Threefish-512.
const PERM8: [usize; 8] = [2, 1, 4, 7, 6, 5, 0, 3];

#[inline(always)]
fn rotl(x: u64, r: u32) -> u64 {
    x.rotate_left(r)
}

/// Threefish-512 block cipher with 64-byte key and 16-byte tweak.
#[derive(Clone)]
struct Threefish512 {
    k: [u64; 9], // k[0..7], k[8] = xor(k[0..7]) ^ C240
    t: [u64; 3], // t[0], t[1], t[2] = t0 ^ t1
}

impl Threefish512 {
    /// Creates a new Threefish-512 instance with the given key and tweak.
    fn new(key_bytes: &[u8; 64], tweak_bytes: &[u8; 16]) -> Self {
        let mut k = [0u64; 9];
        let mut parity = C240;
        for (i, chunk) in key_bytes.chunks_exact(8).enumerate() {
            let ki = u64::from_le_bytes(chunk.try_into().unwrap());
            k[i] = ki;
            parity ^= ki;
        }
        k[8] = parity;
        let t0 = u64::from_le_bytes(tweak_bytes[0..8].try_into().unwrap());
        let t1 = u64::from_le_bytes(tweak_bytes[8..16].try_into().unwrap());
        Threefish512 { k, t: [t0, t1, t0 ^ t1] }
    }

    #[inline(always)]
    fn subkey_word(&self, s: usize, i: usize) -> u64 {
        let base = self.k[(s + i) % 9];
        match i {
            5 => base.wrapping_add(self.t[s % 3]),
            6 => base.wrapping_add(self.t[(s + 1) % 3]),
            7 => base.wrapping_add(s as u64),
            _ => base,
        }
    }

    #[inline(always)]
    fn add_subkey(&self, v: &mut [u64; 8], s: usize) {
        for i in 0..8 {
            v[i] = v[i].wrapping_add(self.subkey_word(s, i));
        }
    }

    /// Encrypts a 64-byte block using Threefish-512 (72 rounds).
    fn encrypt_block(&self, block: &[u8; 64]) -> [u8; 64] {
        let mut v = [0u64; 8];
        for (i, chunk) in block.chunks_exact(8).enumerate() {
            v[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        for d in 0..72 {
            if d % 4 == 0 {
                self.add_subkey(&mut v, d / 4);
            }
            let r = R512[d & 7];
            // MIX on word pairs
            let (mut x0, mut x1) = (v[0], v[1]);
            let y0 = x0.wrapping_add(x1);
            let y1 = rotl(x1, r[0]) ^ y0;
            v[0] = y0;
            v[1] = y1;
            x0 = v[2];
            x1 = v[3];
            let y0 = x0.wrapping_add(x1);
            let y1 = rotl(x1, r[1]) ^ y0;
            v[2] = y0;
            v[3] = y1;
            x0 = v[4];
            x1 = v[5];
            let y0 = x0.wrapping_add(x1);
            let y1 = rotl(x1, r[2]) ^ y0;
            v[4] = y0;
            v[5] = y1;
            x0 = v[6];
            x1 = v[7];
            let y0 = x0.wrapping_add(x1);
            let y1 = rotl(x1, r[3]) ^ y0;
            v[6] = y0;
            v[7] = y1;
            // permutation
            let f = v;
            for i in 0..8 {
                v[i] = f[PERM8[i]];
            }
        }
        self.add_subkey(&mut v, 18);
        let mut out = [0u8; 64];
        for (i, w) in v.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

// ---------- Skein-512 (streaming UBI) ----------

/// Skein-512 hash and MAC implementation using Threefish-512.
mod skein512 {
    use super::Threefish512;

    /// Block size in bytes (64).
    pub const NB: usize = 64;
    /// Type code for key input.
    pub const T_KEY: u8 = 0;
    /// Type code for message input.
    pub const T_MSG: u8 = 48;
    /// Type code for output.
    pub const T_OUT: u8 = 63;
    /// Initial vector for Skein-512-512.
    pub const IV: [u64; 8] = [
        0x4903ADFF749C51CE,
        0x0D95DE399746DF03,
        0x8FD1934127C79BCE,
        0x9A255629FF352CB1,
        0x5DB62599DF6CA7B0,
        0xEABE394CA9D5C3F4,
        0x991112C71A75B523,
        0xAE18A40B660FCC33,
    ];

    #[inline(always)]
    fn u128_to_le_bytes(x: u128) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&(x as u64).to_le_bytes());
        out[8..].copy_from_slice(&((x >> 64) as u64).to_le_bytes());
        out
    }

    #[inline(always)]
    fn ubi_block(
        hi: [u64; 8],
        block: &[u8; NB],
        tcode: u8,
        pos_after: u128,
        first: bool,
        last: bool,
    ) -> [u64; 8] {
        let mut t = (tcode as u128) << 120;
        t += pos_after;
        if first {
            t += 1u128 << 126;
        }
        if last {
            t += 1u128 << 127;
        }
        let mut key_bytes = [0u8; 64];
        for (j, w) in hi.iter().enumerate() {
            key_bytes[j * 8..j * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        let tweak_bytes = u128_to_le_bytes(t);
        let tf = Threefish512::new(&key_bytes, &tweak_bytes);
        let e = tf.encrypt_block(block);
        let mut next = [0u64; 8];
        for j in 0..8 {
            let ej = u64::from_le_bytes(e[j * 8..j * 8 + 8].try_into().unwrap());
            let mj = u64::from_le_bytes(block[j * 8..j * 8 + 8].try_into().unwrap());
            next[j] = ej ^ mj;
        }
        next
    }

    /// Streaming UBI processor for Skein-512.
    pub struct UbiStream {
        hi: [u64; 8],
        tcode: u8,
        first: bool,
        pos: u128,
        buf: [u8; NB],
        buf_len: usize,
    }

    impl UbiStream {
        pub fn new(initial_hi: [u64; 8], tcode: u8) -> Self {
            Self {
                hi: initial_hi,
                tcode,
                first: true,
                pos: 0,
                buf: [0u8; NB],
                buf_len: 0,
            }
        }

        pub fn update(&mut self, mut data: &[u8]) {
            while !data.is_empty() {
                let space = NB - self.buf_len;
                let to_copy = std::cmp::min(space, data.len());
                self.buf[self.buf_len..self.buf_len + to_copy].copy_from_slice(&data[..to_copy]);
                self.buf_len += to_copy;
                data = &data[to_copy..];
                if self.buf_len == NB && !data.is_empty() {
                    let block: [u8; NB] = self.buf;
                    let pos_after = self.pos + NB as u128;
                    self.hi = ubi_block(self.hi, &block, self.tcode, pos_after, self.first, false);
                    self.first = false;
                    self.pos = pos_after;
                    self.buf_len = 0;
                }
            }
        }

        pub fn finalize(self) -> [u64; 8] {
            let mut block = [0u8; NB];
            let take = self.buf_len;
            if take > 0 {
                block[..take].copy_from_slice(&self.buf[..take]);
            }
            let pos_after = self.pos + take as u128;
            ubi_block(self.hi, &block, self.tcode, pos_after, self.first, true)
        }
    }

    #[inline]
    fn ubi_once(hi: [u64; 8], data: &[u8], tcode: u8) -> [u64; 8] {
        let mut s = UbiStream::new(hi, tcode);
        s.update(data);
        s.finalize()
    }

    /// Computes a 64-byte Skein-512-512 hash of the input.
    pub fn hash(bytes: &[u8]) -> [u8; 64] {
        let g1 = ubi_once(IV, bytes, T_MSG);
        let g2 = ubi_once(g1, &0u64.to_le_bytes(), T_OUT);
        let mut out = [0u8; 64];
        for (i, w) in g2.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Streaming Skein-MAC-512 implementation.
    pub struct SkeinMacStream {
        msg: UbiStream,
    }

    impl SkeinMacStream {
        pub fn new(mac_key: &[u8]) -> Self {
            let g1 = ubi_once(IV, mac_key, T_KEY);
            Self { msg: UbiStream::new(g1, T_MSG) }
        }

        pub fn update(&mut self, data: &[u8]) {
            self.msg.update(data);
        }

        pub fn finalize(self) -> [u8; 64] {
            let g2 = self.msg.finalize();
            let g3 = ubi_once(g2, &0u64.to_le_bytes(), T_OUT);
            let mut out = [0u8; 64];
            for (i, w) in g3.iter().enumerate() {
                out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
            }
            out
        }

        pub fn finalize_trunc32(self) -> [u8; 32] {
            let full = self.finalize();
            let mut tag = [0u8; 32];
            tag.copy_from_slice(&full[..32]);
            tag
        }
    }
}

// ---------- Container + streaming AE (CTR + Skein-MAC) ----------

/// Derives encryption and MAC keys using Skein-512 KDF.
fn derive_keys(key_file_bytes: &[u8], salt: &[u8; SALT_LEN]) -> (Zeroizing<[u8; 64]>, Zeroizing<[u8; 64]>) {
    let mut in0 = Vec::with_capacity(6 + SALT_LEN + key_file_bytes.len());
    in0.extend_from_slice(b"TFKDF\0");
    in0.extend_from_slice(salt);
    in0.extend_from_slice(key_file_bytes);
    let k0 = skein512::hash(&in0);
    let mut in1 = Vec::with_capacity(6 + SALT_LEN + key_file_bytes.len());
    in1.extend_from_slice(b"TFKDF\x01");
    in1.extend_from_slice(salt);
    in1.extend_from_slice(key_file_bytes);
    let k1 = skein512::hash(&in1);
    let mut ek = [0u8; 64];
    ek.copy_from_slice(&k0);
    let mut mk = [0u8; 64];
    mk.copy_from_slice(&k1);
    (Zeroizing::new(ek), Zeroizing::new(mk))
}

/// CTR mode using Threefish-512 for streaming encryption/decryption.
struct TfCtr {
    key: [u8; 64],
    t1: u64,
    ctr: u64,
    ks: [u8; 64],
    used: usize,
}

impl TfCtr {
    fn new(enc_key: &[u8; 64], nonce: &[u8; NONCE_LEN]) -> Self {
        let nonce_lo = u64::from_le_bytes(nonce[0..8].try_into().unwrap());
        let nonce_hi = u64::from_le_bytes(nonce[8..16].try_into().unwrap());
        Self {
            key: *enc_key,
            t1: nonce_lo ^ nonce_hi,
            ctr: 0,
            ks: [0u8; 64],
            used: 64,
        }
    }

    fn refill(&mut self) {
        let mut tweak = [0u8; 16];
        tweak[0..8].copy_from_slice(&self.ctr.to_le_bytes());
        tweak[8..16].copy_from_slice(&self.t1.to_le_bytes());
        let tf = Threefish512::new(&self.key, &tweak);
        self.ks = tf.encrypt_block(&[0u8; 64]);
        self.ctr = self.ctr.wrapping_add(1);
        self.used = 0;
    }

    fn xor_into(&mut self, input: &[u8], out: &mut [u8]) {
        let mut i = 0usize;
        while i < input.len() {
            if self.used == 64 {
                self.refill();
            }
            let take = std::cmp::min(64 - self.used, input.len() - i);
            for j in 0..take {
                out[i + j] = input[i + j] ^ self.ks[self.used + j];
            }
            i += take;
            self.used += take;
        }
    }
}

/// Generates a temporary file path near the target.
fn temp_path_near(target: &Path) -> PathBuf {
    let dir = target.parent().unwrap_or(Path::new("."));
    let base = target.file_name().unwrap_or_default().to_string_lossy();
    let mut rnd = [0u8; 8];
    OsRng.fill_bytes(&mut rnd);
    dir.join(format!(".{}.{}.tf512.tmp", base, encode(rnd)))
}

/// Performs atomic file replacement (cross-platform).
fn atomic_replace(temp: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        fs::remove_file(dst)?;
    }
    fs::rename(temp, dst)?;
    Ok(())
}

/// Reads and validates the key file.
fn read_key_for(key_path: &Path) -> Result<Zeroizing<Vec<u8>>> {
    let bytes = fs::read(key_path).map_err(|e| {
        anyhow!(
            "Failed to read key file: {}\nExpected at: {}\nGenerate one with `threefish512-app gen-key`",
            e,
            key_path.display()
        )
    })?;
    if bytes.len() < 16 {
        return Err(anyhow!("Key file must be at least 16 bytes (recommend 64 bytes)"));
    }
    if bytes.len() > 1024 {
        return Err(anyhow!("Key file too large (max 1024 bytes, got {})", bytes.len()));
    }
    Ok(Zeroizing::new(bytes))
}

/// Encrypts a file in-place with Threefish-512 CTR and Skein-MAC.
/// Uses a nonce store to prevent nonce reuse for the same key.
fn encrypt_in_place_streaming(path: &Path, key_path: &Path) -> Result<()> {
    let key_file = read_key_for(key_path)?;
    let mut nonce_store = NonceStore::new(key_path)?;
    let mut salt = [0u8; SALT_LEN];
    OsRng.fill_bytes(&mut salt);
    let nonce = nonce_store.generate_unique_nonce()?;
    let (enc_key, mac_key) = derive_keys(&key_file, &salt);
    let mut mac = skein512::SkeinMacStream::new(&*mac_key);
    let mut ctr = TfCtr::new(&enc_key, &nonce);

    let meta = fs::metadata(path)?;
    let orig_len = meta.len() as u64;
    let mut reader = BufReader::new(File::open(path)?);
    reader.get_ref().lock_exclusive()?;

    let tmp = temp_path_near(path);
    let mut out = OpenOptions::new().create_new(true).write(true).open(&tmp)?;
    out.lock_exclusive()?;

    let mut header = Vec::with_capacity(MAGIC.len() + SALT_LEN + NONCE_LEN + 8);
    header.extend_from_slice(&MAGIC);
    header.extend_from_slice(&salt);
    header.extend_from_slice(&nonce);
    header.extend_from_slice(&orig_len.to_le_bytes());
    out.write_all(&header)?;
    mac.update(&header);

    let mut inbuf = vec![0u8; CHUNK];
    let mut outbuf = vec![0u8; CHUNK];
    loop {
        let n = reader.read(&mut inbuf)?;
        if n == 0 {
            break;
        }
        ctr.xor_into(&inbuf[..n], &mut outbuf[..n]);
        mac.update(&outbuf[..n]);
        out.write_all(&outbuf[..n])?;
    }

    let tag = mac.finalize_trunc32();
    out.write_all(&tag)?;
    out.sync_all()?;

    drop(out);
    drop(reader);

    // Save nonce store before replacing file
    nonce_store.save()?;

    match atomic_replace(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Decrypts a file in-place with Threefish-512 CTR and Skein-MAC verification.
fn decrypt_in_place_streaming(path: &Path, key_path: &Path) -> Result<()> {
    let total_len = fs::metadata(path)?.len() as usize;
    if total_len < MAGIC.len() + SALT_LEN + NONCE_LEN + 8 + TAG_LEN {
        return Err(anyhow!("File too short or not a TF512 container"));
    }

    let mut f = File::open(path)?;
    f.lock_exclusive()?;
    let mut header = [0u8; 8 + SALT_LEN + NONCE_LEN + 8];
    f.read_exact(&mut header)?;
    if &header[..MAGIC.len()] != MAGIC {
        return Err(anyhow!("Invalid magic bytes (not a TF512v1 container)"));
    }

    let mut idx = MAGIC.len();
    let salt = <[u8; SALT_LEN]>::try_from(&header[idx..idx + SALT_LEN]).unwrap();
    idx += SALT_LEN;
    let nonce = <[u8; NONCE_LEN]>::try_from(&header[idx..idx + NONCE_LEN]).unwrap();
    idx += NONCE_LEN;
    let orig_len = u64::from_le_bytes(header[idx..idx + 8].try_into().unwrap());
    let header_len = header.len();
    let ct_len = total_len
        .checked_sub(header_len + TAG_LEN)
        .ok_or_else(|| anyhow!("Truncated file"))?;

    let key_file = read_key_for(key_path)?;
    let (enc_key, mac_key) = derive_keys(&key_file, &salt);
    let mut mac = skein512::SkeinMacStream::new(&*mac_key);
    mac.update(&header);

    let tmp = temp_path_near(path);
    let mut out = OpenOptions::new().create_new(true).write(true).open(&tmp)?;
    out.lock_exclusive()?;

    let mut ctr = TfCtr::new(&enc_key, &nonce);
    let mut remaining = ct_len;
    let mut inbuf = vec![0u8; CHUNK];
    let mut ptbuf = vec![0u8; CHUNK];
    while remaining > 0 {
        let to_read = std::cmp::min(remaining, CHUNK);
        f.read_exact(&mut inbuf[..to_read])?;
        mac.update(&inbuf[..to_read]);
        ctr.xor_into(&inbuf[..to_read], &mut ptbuf[..to_read]);
        out.write_all(&ptbuf[..to_read])?;
        remaining -= to_read;
    }

    let mut got_tag = [0u8; TAG_LEN];
    f.read_exact(&mut got_tag)?;
    let exp_tag = mac.finalize_trunc32();
    if !bool::from(got_tag.ct_eq(&exp_tag)) {
        drop(out);
        let _ = fs::remove_file(&tmp);
        return Err(anyhow!("Authentication failed (incorrect key or corrupted file)"));
    }

    if ct_len as u64 != orig_len {
        drop(out);
        let _ = fs::remove_file(&tmp);
        return Err(anyhow!(
            "Length mismatch (header claims {}, ciphertext is {} bytes)",
            orig_len,
            ct_len
        ));
    }

    out.sync_all()?;
    drop(out);
    drop(f);

    atomic_replace(&tmp, path)?;
    Ok(())
}

// ---------- CLI ----------

#[derive(Parser)]
#[command(name = "threefish512-app")]
#[command(about = "Threefish-512 file locker (in-place, streaming) + Skein self-test")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Atomically encrypts a file in place.
    /// Requires a key file (default: key.key in the same directory).
    Lock {
        #[arg(value_name = "PATH")]
        path: PathBuf,
        #[arg(long, value_name = "KEY_PATH", default_value = "key.key")]
        key: PathBuf,
    },
    /// Atomically decrypts a file in place.
    /// Requires a key file (default: key.key in the same directory).
    Unlock {
        #[arg(value_name = "PATH")]
        path: PathBuf,
        #[arg(long, value_name = "KEY_PATH", default_value = "key.key")]
        key: PathBuf,
    },
    /// Creates a random key file (default: key.key, 64 bytes).
    GenKey {
        #[arg(long, value_name = "PATH", default_value = "key.key")]
        out: PathBuf,
        #[arg(long, default_value_t = 64)]
        size: usize,
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// Runs Skein-512-512 self-test with known vector (0xFF).
    SelfTest,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Lock { path, key } => {
            encrypt_in_place_streaming(&path, &key)?;
            println!("Locked: {}", path.display());
        }
        Commands::Unlock { path, key } => {
            decrypt_in_place_streaming(&path, &key)?;
            println!("Unlocked: {}", path.display());
        }
        Commands::GenKey { out, size, force } => {
            if out.exists() && !force {
                return Err(anyhow!("{} exists; use --force to overwrite", out.display()));
            }
            if size < 16 {
                return Err(anyhow!("Key size must be at least 16 bytes"));
            }
            if size > 1024 {
                return Err(anyhow!("Key size too large (max 1024 bytes)"));
            }
            let mut buf = vec![0u8; size];
            OsRng.fill_bytes(&mut buf);
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&out)?;
            f.lock_exclusive()?;
            f.write_all(&buf)?;
            f.sync_all()?;
            println!("Wrote {} ({} bytes)", out.display(), size);
        }
        Commands::SelfTest => {
            let got = skein512::hash(&[0xFF]);
            let expected_hex = concat!(
                "71b7bce6fe6452227b9ced6014249e5b",
                "f9a9754c3ad618ccc4e0aae16b316cc8",
                "ca698d864307ed3e80b6ef1570812ac5",
                "272dc409b5a012df2a579102f340617a"
            );
            let expected = hex::decode(expected_hex).unwrap();
            if !bool::from(got.ct_eq(&expected)) {
                return Err(anyhow!(
                    "Self-test failed\nGot: {}\nExpected: {}",
                    encode(got),
                    expected_hex
                ));
            }
            println!("Skein-512 self-test: OK");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn skein_vector_ff() {
        let got = skein512::hash(&[0xFF]);
        let expected = hex::decode(concat!(
            "71b7bce6fe6452227b9ced6014249e5b",
            "f9a9754c3ad618ccc4e0aae16b316cc8",
            "ca698d864307ed3e80b6ef1570812ac5",
            "272dc409b5a012df2a579102f340617a"
        ))
        .unwrap();
        assert!(bool::from(got.ct_eq(&expected)), "Skein-512 hash mismatch");
    }

    #[test]
    fn threefish_round_trip() {
        let key = [0u8; 64];
        let tweak = [0u8; 16];
        let tf = Threefish512::new(&key, &tweak);
        let plaintext = [0xFF; 64];
        let ciphertext = tf.encrypt_block(&plaintext);
        let tf_dec = Threefish512::new(&key, &tweak);
        let decrypted = tf_dec.encrypt_block(&ciphertext); // CTR is symmetric
        assert!(bool::from(plaintext.ct_eq(&decrypted)), "Threefish-512 round-trip failed");
    }

    #[test]
    fn encrypt_decrypt_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let key_path = dir.path().join("key.key");

        // Create a test file
        let mut f = File::create(&file_path).unwrap();
        let data = b"Hello, Threefish-512!";
        f.write_all(data).unwrap();
        f.sync_all().unwrap();

        // Create a key file
        let mut key_file = File::create(&key_path).unwrap();
        let key_data = [0xAA; 64];
        key_file.write_all(&key_data).unwrap();
        key_file.sync_all().unwrap();

        // Encrypt
        encrypt_in_place_streaming(&file_path, &key_path).unwrap();

        // Decrypt
        decrypt_in_place_streaming(&file_path, &key_path).unwrap();

        // Verify
        let decrypted = fs::read(&file_path).unwrap();
        assert!(bool::from(decrypted.ct_eq(data)), "Round-trip encryption/decryption failed");
    }

    #[test]
    fn nonce_store_prevents_reuse() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = dir.path().join("key.key");
        let mut key_file = File::create(&key_path).unwrap();
        key_file.write_all(&[0xAA; 64]).unwrap();
        key_file.sync_all().unwrap();

        let mut store = NonceStore::new(&key_path).unwrap();
        let nonce1 = store.generate_unique_nonce().unwrap();
        store.save().unwrap();

        let mut store = NonceStore::new(&key_path).unwrap();
        let nonce2 = store.generate_unique_nonce().unwrap();
        assert!(!bool::from(nonce1.ct_eq(&nonce2)), "Nonces must be unique");
    }
}