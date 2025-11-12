# Threefish-1024 File Encryption CLI

A command-line tool for secure file encryption and decryption using the Threefish-1024 block cipher in Counter (CTR) mode. This Rust-based app supports in-place file processing, automatically detects whether to encrypt or decrypt based on a magic header, and expects a 128-byte key from a `key.bin` file in the executable’s directory.

## Features

- **Threefish-1024 Cipher**: Uses a 1024-bit block cipher for high-security encryption.
- **CTR Mode**: Stream-like encryption, suitable for files of any size.
- **In-Place Processing**: Overwrites files atomically to encrypt or decrypt.
- **Auto-Detection**: Identifies plaintext vs. encrypted files using a `TF1024ENC` magic header and embedded 16-byte IV.
- **Key Management**: Loads a 128-byte key from `key.bin` in the executable’s directory; includes a command to generate a random key.
- **Simple CLI**: Minimal commands: `threefish_encrypt process <file_name>` or `threefish_encrypt generate-key`.
- **Same-Directory Restriction**: Input files and `key.bin` must be in the executable’s directory.

## Installation

1. Ensure [Rust](https://www.rust-lang.org/) and Cargo are installed.
2. Create a new Rust project or clone this repository:
   ```bash
   cargo new threefish_encrypt
   cd threefish_encrypt
   ```
3. Replace `Cargo.toml` with:
   ```toml
   [package]
   name = "threefish_encrypt"
   version = "0.1.0"
   edition = "2021"

   [dependencies]
   threefish = "0.5"
   cipher = "0.4"
   clap = { version = "4.5", features = ["derive"] }
   hex = "0.4"
   anyhow = "1.0"
   rand = "0.8"
   ```
4. Replace `src/main.rs` with the project’s source code (see repository or documentation).
5. Build the project:
   ```bash
   cargo build --release
   ```
   The executable will be in `target/release/threefish_encrypt`.

## Usage

The CLI requires input files and the key file (`key.bin`) to be in the same directory as the executable (e.g., `target/release/`).

### Generate a Key
Create a random 128-byte key file (`key.bin`):
```bash
cd target/release
./threefish_encrypt generate-key
```
This generates `key.bin` in the executable’s directory. Optionally, specify a different key file name:
```bash
./threefish_encrypt generate-key --output mykey.bin
```

### Encrypt or Decrypt a File
Process a file (encrypt if plaintext, decrypt if encrypted):
```bash
./threefish_encrypt process test.txt
```
- If `test.txt` is plaintext, it’s encrypted in-place with a random IV, prepending a `TF1024ENC` header and the IV.
- If `test.txt` is encrypted (starts with `TF1024ENC`), it’s decrypted in-place using the IV from the header.
- Requires `key.bin` in the executable’s directory.

**Example**:
```bash
cd target/release
echo "Hello, Threefish!" > test.txt
./threefish_encrypt generate-key
./threefish_encrypt process test.txt  # Encrypts test.txt
./threefish_encrypt process test.txt  # Decrypts test.txt
cat test.txt  # Outputs: Hello, Threefish!
```

## Security Considerations

- **IV Uniqueness**: The app generates a random 16-byte IV for each encryption, stored in the file header, ensuring CTR mode security. Never manually reuse IVs.
- **Key Management**: Store `key.bin` securely. The 128-byte key (1024 bits) provides a high security margin but must be protected.
- **Integrity**: The current implementation lacks authentication (e.g., HMAC). For production use, consider adding integrity checks to detect tampering.
- **Threefish-1024**: A robust cipher with no known practical attacks, though less scrutinized than AES. Suitable for high-security applications.
- **Directory Restriction**: Files and `key.bin` must be in the executable’s directory, reducing path-related vulnerabilities but requiring manual file placement.

## Technical Details

- **Cipher**: Threefish-1024 (1024-bit block, 128-byte key, 16-byte tweak/IV, 80 rounds).
- **Mode**: CTR mode with a 64-bit counter, XORing keystream with data.
- **Dependencies**:
  - `threefish = "0.5"`: Implements Threefish-1024.
  - `cipher = "0.4"`: Provides block cipher traits.
  - `clap = "4.5"`: CLI argument parsing.
  - `rand = "0.8"`: Random IV and key generation.
  - `anyhow = "1.0"`: Error handling.
- **File Format**: Encrypted files start with a 9-byte magic header (`TF1024ENC`) followed by a 16-byte IV, then the ciphertext.
- **Limitations**: Loads files into memory, which may be inefficient for large files. Streaming I/O could be added for better performance.

## Troubleshooting

- **Key File Missing**: Ensure `key.bin` exists in the executable’s directory. Run `generate-key` to create it.
- **Invalid Key Size**: `key.bin` must be exactly 128 bytes. Regenerate if invalid.
- **File Not Found**: Place input files (e.g., `test.txt`) in the executable’s directory (e.g., `target/release/`).
- **Permissions**: Ensure the executable’s directory is writable for temp files and key generation.
- **Compilation Issues**: Run `cargo clean` and `cargo update` to resolve dependency conflicts. Verify `Cargo.toml` and `src/main.rs` match the provided versions.

## Contributing

Feel free to submit issues or pull requests to the project repository for bug fixes, feature additions (e.g., HMAC, streaming I/O, password support), or improvements.

## License

This project is unlicensed or can be licensed under MIT, at your discretion.

*Last updated: October 2025*