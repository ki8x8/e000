use anyhow::{anyhow, Context, Result};
use chacha20poly1305::{XChaCha20Poly1305, XNonce, aead::{Aead, KeyInit, Payload}};
use clap::{Parser, ValueEnum};
use fs2::FileExt;
use rand::rngs::OsRng;
use rand::RngCore;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};
use log::{info, trace, warn};
use zeroize::Zeroize;

// ---------- Constants ----------
const MAGIC: [u8; 8] = *b"SPSv2\0\0\0"; // Version 2 magic
const BASE_LEN: usize = 16; // Base for nonce derivation
const TAG_LEN: usize = 16; // Poly1305 tag
const CHUNK: usize = 8 << 20; // 8 MiB chunk size
const KEY_LEN: usize = 32; // 256-bit key
const HEADER_LEN: usize = MAGIC.len() + BASE_LEN + 8;

// ---------- File Utilities ----------
fn temp_path_near(target: &Path) -> PathBuf {
    let base = target.file_name().unwrap_or_default().to_string_lossy();
    let mut rnd = [0u8; 8];
    OsRng.fill_bytes(&mut rnd);
    PathBuf::from(format!(".{}.{}.sps.tmp", base, hex::encode(rnd)))
}

fn atomic_replace(temp: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        fs::remove_file(dst).context(format!("Failed to remove destination file {}", dst.display()))?;
    }
    fs::rename(temp, dst).context(format!("Failed to rename {} to {}", temp.display(), dst.display()))?;
    Ok(())
}

// ---------- Encryption ----------
fn encrypt_file(input_path: &Path, key: &mut [u8; KEY_LEN], verbose: bool) -> Result<()> {
    let mut file = File::open(input_path).context(format!("Failed to open input file {}", input_path.display()))?;
    file.lock_exclusive()?;
    let orig_len = file.metadata()?.len();
    if orig_len == 0 {
        warn!("Encrypting empty file: {}", input_path.display());
    }
    let mut base = [0u8; BASE_LEN];
    OsRng.fill_bytes(&mut base);
    let aead = XChaCha20Poly1305::new_from_slice(key).map_err(|e| anyhow!("Invalid key length: {}", e))?;
    let mut reader = BufReader::new(&mut file);
    let tmp = temp_path_near(input_path);
    let mut out_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&tmp)
        .context(format!("Failed to create temp file {}", tmp.display()))?;
    out_file.lock_exclusive()?;
    let mut writer = BufWriter::new(&mut out_file);
    // Write header
    let header: [u8; HEADER_LEN] = {
        let mut h = [0u8; HEADER_LEN];
        h[..MAGIC.len()].copy_from_slice(&MAGIC);
        h[MAGIC.len()..MAGIC.len() + BASE_LEN].copy_from_slice(&base);
        h[MAGIC.len() + BASE_LEN..].copy_from_slice(&orig_len.to_le_bytes());
        h
    };
    writer.write_all(&header)?;
    // Encrypt in chunks
    let mut inbuf = vec![0u8; CHUNK];
    let mut processed = 0u64;
    let mut chunk_idx = 0u64;
    while processed < orig_len {
        let n = reader.read(&mut inbuf[..])?;
        if n == 0 {
            break;
        }
        let nonce = {
            let mut nb = [0u8; 24];
            nb[..BASE_LEN].copy_from_slice(&base);
            nb[BASE_LEN..].copy_from_slice(&chunk_idx.to_le_bytes());
            nb
        };
        let aad = {
            let mut a = [0u8; HEADER_LEN + 8];
            a[..HEADER_LEN].copy_from_slice(&header);
            a[HEADER_LEN..].copy_from_slice(&chunk_idx.to_le_bytes());
            a
        };
        let payload = Payload { msg: &inbuf[..n], aad: &aad };
        let ciphertext = aead.encrypt(XNonce::from_slice(&nonce), payload)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;
        writer.write_all(&ciphertext)?;
        processed += n as u64;
        chunk_idx += 1;
        if verbose {
            trace!("Encrypted chunk {} ({} bytes)", chunk_idx, n);
        }
    }
    if processed != orig_len {
        drop(writer);
        let _ = fs::remove_file(&tmp);
        return Err(anyhow!("Incomplete read during encryption (processed {} of {})", processed, orig_len));
    }
    writer.flush()?;
    drop(writer);
    atomic_replace(&tmp, input_path)?;
    Ok(())
}

// ---------- Decryption ----------
fn decrypt_file(input_path: &Path, key: &mut [u8; KEY_LEN], verbose: bool) -> Result<()> {
    let mut file = File::open(input_path).context(format!("Failed to open input file {}", input_path.display()))?;
    file.lock_exclusive()?;
    let total_len = file.metadata()?.len();
    if total_len < HEADER_LEN as u64 {
        return Err(anyhow!("File too short for SPS container ({} bytes)", total_len));
    }
    let mut header = [0u8; HEADER_LEN];
    file.read_exact(&mut header)?;
    if &header[..MAGIC.len()] != MAGIC {
        return Err(anyhow!("Invalid magic bytes"));
    }
    let base: [u8; BASE_LEN] = header[MAGIC.len()..MAGIC.len() + BASE_LEN]
        .try_into()
        .map_err(|_| anyhow!("Invalid base nonce"))?;
    let orig_len = u64::from_le_bytes(
        header[MAGIC.len() + BASE_LEN..]
            .try_into()
            .map_err(|_| anyhow!("Invalid original length"))?,
    );
    let aead = XChaCha20Poly1305::new_from_slice(key).map_err(|e| anyhow!("Invalid key length: {}", e))?;
    let tmp = temp_path_near(input_path);
    let mut out_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&tmp)
        .context(format!("Failed to create temp file {}", tmp.display()))?;
    out_file.lock_exclusive()?;
    let mut writer = BufWriter::new(&mut out_file);
    let mut inbuf = vec![0u8; CHUNK + TAG_LEN];
    let mut processed = 0u64;
    let mut chunk_idx = 0u64;
    while processed < orig_len {
        let expected_plaintext_len = std::cmp::min((orig_len - processed) as usize, CHUNK);
        if expected_plaintext_len == 0 {
            break;
        }
        let to_read = expected_plaintext_len + TAG_LEN;
        let n = file.read(&mut inbuf[..to_read])?;
        if n != to_read {
            return Err(anyhow!("Incomplete read during decryption (expected {}, got {})", to_read, n));
        }
        let nonce = {
            let mut nb = [0u8; 24];
            nb[..BASE_LEN].copy_from_slice(&base);
            nb[BASE_LEN..].copy_from_slice(&chunk_idx.to_le_bytes());
            nb
        };
        let aad = {
            let mut a = [0u8; HEADER_LEN + 8];
            a[..HEADER_LEN].copy_from_slice(&header);
            a[HEADER_LEN..].copy_from_slice(&chunk_idx.to_le_bytes());
            a
        };
        let payload = Payload { msg: &inbuf[..to_read], aad: &aad };
        let plaintext = aead.decrypt(XNonce::from_slice(&nonce), payload)
            .map_err(|_| anyhow!("Decryption failed: wrong key or corrupted file (chunk {})", chunk_idx))?;
        if plaintext.len() != expected_plaintext_len {
            return Err(anyhow!("Decrypted chunk length mismatch (expected {}, got {})", expected_plaintext_len, plaintext.len()));
        }
        writer.write_all(&plaintext)?;
        processed += expected_plaintext_len as u64;
        chunk_idx += 1;
        if verbose {
            trace!("Decrypted chunk {} ({} bytes)", chunk_idx, expected_plaintext_len);
        }
    }
    writer.flush()?;
    let decrypted_len = writer.get_ref().metadata()?.len();
    if decrypted_len != orig_len {
        return Err(anyhow!("Final length mismatch (expected {}, got {})", orig_len, decrypted_len));
    }
    let current_pos = file.stream_position()?;
    if current_pos != total_len {
        drop(writer);
        let _ = fs::remove_file(&tmp);
        return Err(anyhow!("Extra data after ciphertext ({} extra bytes)", total_len - current_pos));
    }
    drop(writer);
    atomic_replace(&tmp, input_path)?;
    Ok(())
}

// ---------- Key Management ----------
fn load_key(key_path: &Path) -> Result<[u8; KEY_LEN]> {
    let metadata = fs::metadata(key_path)
        .context(format!("Failed to access key file {}", key_path.display()))?;
    if !metadata.is_file() {
        return Err(anyhow!("Key file {} is not a regular file", key_path.display()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = metadata.permissions().mode();
        if mode & 0o077 != 0 {
            warn!("Key file {} has overly permissive permissions (mode {:o})", key_path.display(), mode);
        }
    }
    let mut key_bytes = fs::read(key_path)
        .context(format!("Failed to read key file {}", key_path.display()))?;
    if key_bytes.len() != KEY_LEN {
        key_bytes.zeroize();
        return Err(anyhow!("Invalid key file length: expected {} bytes, got {}", KEY_LEN, key_bytes.len()));
    }
    let mut key = [0u8; KEY_LEN];
    key.copy_from_slice(&key_bytes[..KEY_LEN]);
    key_bytes.zeroize();
    Ok(key)
}

// ---------- CLI ----------
#[derive(ValueEnum, Clone, Copy)]
enum ForceMode {
    Encrypt,
    Decrypt,
}

#[derive(Parser)]
#[command(name = "sps")]
#[command(about = "Simple file encryption/decryption with XChaCha20-Poly1305.\nFiles must be in the current directory (no paths).")]
struct Cli {
    /// The filename to process
    filename: String,
    /// Path to the key file (default: key.key)
    #[arg(long, default_value = "key.key")]
    key_file: PathBuf,
    /// Force operation (encrypt or decrypt)
    #[arg(long, value_enum)]
    force: Option<ForceMode>,
    /// Generate a new key file (overwrites if exists)
    #[arg(long)]
    generate_key: bool,
    /// Enable verbose logging
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::builder()
        .filter_level(if cli.verbose { log::LevelFilter::Trace } else { log::LevelFilter::Warn })
        .init();

    if cli.generate_key {
        let mut key = [0u8; KEY_LEN];
        OsRng.fill_bytes(&mut key);
        fs::write(&cli.key_file, &key).context(format!("Failed to write key file {}", cli.key_file.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&cli.key_file, fs::Permissions::from_mode(0o600))
                .context(format!("Failed to set permissions on key file {}", cli.key_file.display()))?;
        }
        key.zeroize();
        println!("Generated new key in {}", cli.key_file.display());
        return Ok(());
    }

    let filename = cli.filename.trim();
    if filename.contains('/') || filename.contains('\\') || filename.contains("..") || filename.starts_with('.') {
        return Err(anyhow!("Invalid filename: paths and hidden files not allowed"));
    }
    let path = PathBuf::from(filename);
    if !path.exists() || !path.is_file() {
        return Err(anyhow!("File does not exist or is not a regular file: {}", filename));
    }

    let mut key = load_key(&cli.key_file)?;
    let mut file = File::open(&path)?;
    let mut magic_buf = [0u8; MAGIC.len()];
    let read_bytes = file.read(&mut magic_buf)?;
    let has_magic = read_bytes == MAGIC.len() && magic_buf == MAGIC;
    drop(file);

    let should_encrypt = match cli.force {
        Some(ForceMode::Encrypt) => true,
        Some(ForceMode::Decrypt) => false,
        None => !has_magic,
    };

    if should_encrypt {
        info!("Encrypting file: {}", path.display());
        encrypt_file(&path, &mut key, cli.verbose)?;
    } else {
        info!("Decrypting file: {}", path.display());
        decrypt_file(&path, &mut key, cli.verbose)?;
    }

    key.zeroize();
    println!("ok");
    Ok(())
}