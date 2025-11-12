use anyhow::{Context, Result};
use cipher::{BlockEncrypt, generic_array::GenericArray};
use clap::{Parser, Subcommand};
use rand::RngCore;
use std::fs::{rename, write, File};
use std::io::Read;
use threefish::Threefish1024;

const MAGIC_HEADER: &[u8] = b"TF1024ENC"; // 9 bytes magic
const IV_SIZE: usize = 16;
const HEADER_SIZE: usize = MAGIC_HEADER.len() + IV_SIZE; // 9 + 16 = 25 bytes
const KEY_FILE: &str = "key.bin"; // Expected key file name

#[derive(Parser)]
#[command(name = "threefish_encrypt", about = "CLI for Threefish-1024 file encryption/decryption with auto-detect and in-place overwrite")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Process file: encrypt if plaintext, decrypt if encrypted (in-place)
    Process {
        /// File name (in same directory as executable)
        file: String,
    },
    /// Generate a random 128-byte key file (key.bin in exe dir)
    GenerateKey {
        /// Output key file name (defaults to key.bin)
        #[arg(short, long, default_value = "key.bin")]
        output: String,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Get executable directory
    let exe_dir = std::env::current_exe()?
        .parent()
        .context("Failed to get exe dir")?
        .to_path_buf();

    match args.command {
        Command::Process { file } => {
            let file_path = exe_dir.join(&file);
            let temp_path = exe_dir.join(format!("{}.tmp", file_path.file_name().unwrap().to_str().unwrap()));

            // Read key from key.bin
            let key_path = exe_dir.join(KEY_FILE);
            let mut key_bytes = vec![];
            File::open(&key_path)
                .context(format!("Failed to open key file: {}", key_path.display()))?
                .read_to_end(&mut key_bytes)
                .context("Failed to read key file")?;
            if key_bytes.len() != 128 {
                return Err(anyhow::anyhow!("Key file {} must be exactly 128 bytes", key_path.display()));
            }
            let key_array: [u8; 128] = key_bytes.try_into().map_err(|_| anyhow::anyhow!("Key conversion error"))?;

            // Read input file
            let mut data = vec![];
            File::open(&file_path)?.read_to_end(&mut data).context("Failed to read file")?;

            let (output, is_decrypt) = if data.starts_with(MAGIC_HEADER) {
                // Decrypt: Extract IV and ciphertext
                if data.len() < HEADER_SIZE {
                    return Err(anyhow::anyhow!("Invalid encrypted file (too short)"));
                }
                let iv_array: [u8; 16] = data[MAGIC_HEADER.len()..HEADER_SIZE].try_into()?;
                let ciphertext = &data[HEADER_SIZE..];
                (process_ctr(&key_array, &iv_array, ciphertext)?, true)
            } else {
                // Encrypt: Generate random IV, process, prepend header
                let mut iv_array = [0u8; 16];
                rand::thread_rng().fill_bytes(&mut iv_array);
                let ciphertext = process_ctr(&key_array, &iv_array, &data)?;
                let mut output = MAGIC_HEADER.to_vec();
                output.extend_from_slice(&iv_array);
                output.extend_from_slice(&ciphertext);
                (output, false)
            };

            // Write to temp file
            write(&temp_path, &output).context("Failed to write temp file")?;

            // Atomic rename
            rename(&temp_path, &file_path).context("Failed to rename temp file")?;

            println!("File {} successfully {}!", file_path.display(), if is_decrypt { "decrypted" } else { "encrypted" });
        }
        Command::GenerateKey { output } => {
            let key_path = exe_dir.join(output);
            let mut key = [0u8; 128];
            rand::thread_rng().fill_bytes(&mut key);
            write(&key_path, &key).context("Failed to write key file")?;
            println!("Generated key file: {}", key_path.display());
        }
    }
    Ok(())
}

fn process_ctr(key_array: &[u8; 128], iv_array: &[u8; 16], data: &[u8]) -> Result<Vec<u8>> {
    // Threefish-1024 instance with key and tweak (IV)
    let cipher = Threefish1024::new_with_tweak(key_array, iv_array);

    // CTR mode: Generate keystream, XOR with data
    let block_size = 128;
    let num_blocks = (data.len() + block_size - 1) / block_size;
    let mut output = vec![0u8; data.len()];
    let mut counter: u64 = 0;

    for i in 0..num_blocks {
        let mut ctr_block = [0u8; 128];
        ctr_block[0..8].copy_from_slice(&counter.to_le_bytes());
        let mut ctr_generic = GenericArray::from_mut_slice(&mut ctr_block);

        <Threefish1024 as BlockEncrypt>::encrypt_block(&cipher, &mut ctr_generic);

        let start = i * block_size;
        let end = std::cmp::min(start + block_size, data.len());
        for j in start..end {
            output[j] = data[j] ^ ctr_block[j - start];
        }

        counter += 1;
    }

    Ok(output)
}