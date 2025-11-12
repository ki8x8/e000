# SPS: Simple File Encryption Tool

SPS (Simple Protection System) is a command-line tool written in Rust for encrypting and decrypting files using the XChaCha20-Poly1305 authenticated encryption algorithm. It is designed to be secure, efficient, and easy to use, processing files in the current working directory with a 256-bit key stored in a file (default: `key.key`).

## Features
- **Secure Encryption**: Uses XChaCha20-Poly1305 for strong confidentiality and integrity.
- **Chunked Processing**: Handles large files efficiently in 8 MiB chunks.
- **Automatic Operation Detection**: Encrypts or decrypts based on the file's magic header (`SPSv2`).
- **Force Options**: Supports `--force encrypt` or `--force decrypt` to override default behavior.
- **Key Management**: Generates and uses a secure 256-bit key, with memory zeroing for safety.
- **File Locking**: Prevents concurrent access issues during processing.
- **Verbose Logging**: Optional detailed output for debugging.

## Installation

1. **Install Rust**: Follow the instructions at [rustup.rs](https://www.rust-lang.org/tools/install).
2. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd sps
   ```
3. **Build the Project**:
   ```bash
   cargo build --release
   ```
   The executable will be in `target/release/sps`.
4. **Add to PATH** (optional): Copy `target/release/sps` to a directory in your PATH (e.g., `/usr/local/bin`).

## Dependencies
SPS relies on the following Rust crates:
- `chacha20poly1305`: XChaCha20-Poly1305 encryption.
- `clap`: Command-line argument parsing.
- `zeroize`: Secure memory zeroing for keys.
- `fs2`: File locking.
- `rand`, `log`, `env_logger`, `hex`, `anyhow`: Utilities for randomness, logging, and error handling.

See `Cargo.toml` for full details.

## Usage

SPS operates on files in the current directory (no paths allowed). It automatically detects whether to encrypt or decrypt based on the presence of the `SPSv2` magic header.

### Command Syntax
```bash
sps [OPTIONS] <filename>
```

### Options
- `--key-file <PATH>`: Path to the key file (default: `key.key`).
- `--force [encrypt|decrypt]`: Force encryption or decryption, overriding automatic detection.
- `--generate-key`: Generate a new 256-bit key file (overwrites existing).
- `--verbose`: Enable detailed logging.

### Examples
```bash
# Generate a new key
sps --generate-key

# Encrypt a file
sps document.txt

# Decrypt a file
sps document.txt

# Use a custom key file
sps --key-file mykey.key document.txt

# Force encryption (e.g., for double encryption)
sps --force encrypt document.txt

# Force decryption (e.g., for files without magic header)
sps --force decrypt document.txt

# Enable verbose logging
sps --verbose document.txt
```

## Cryptographic Design

SPS uses [XChaCha20-Poly1305](https://libsodium.gitbook.io/doc/secret-key_cryptography/aead#xchacha20-poly1305-ietf), a secure AEAD cipher combining ChaCha20 (stream cipher) and Poly1305 (authentication).

### Key Components
- **Key**: 256-bit key stored in a file (e.g., `key.key`).
- **Nonce**: 192-bit (24-byte) nonces, unique per chunk (16-byte random base + 8-byte chunk index).
- **Chunking**: Files are processed in 8 MiB chunks, each with its own nonce and 16-byte Poly1305 tag.
- **AAD**: Includes the 32-byte header (magic, nonce base, original length) and chunk index, preventing reordering.
- **File Format**: 
  - 32-byte header: 8-byte magic (`SPSv2`), 16-byte nonce base, 8-byte original length.
  - Followed by chunks: encrypted data (up to 8 MiB) + 16-byte tag.

### Security Properties
- Unique nonces per chunk prevent reuse vulnerabilities.
- AAD ensures chunk order and header integrity.
- Secure randomness via `OsRng`.
- Memory zeroing of keys using `zeroize`.
- File locking to prevent concurrent access.

## Security Considerations
- **Key Management**: Store the key file securely. SPS sets 0600 permissions on Unix when generating keys and warns if permissions are too lax.
- **File Overwrites**: SPS overwrites the input file. Back up critical data.
- **Wrong Key**: Decryption with an incorrect key fails with a clear error.
- **No Password Support**: Uses raw keys, not passwords. Consider Argon2 for password-based encryption (future feature).
- **Temporary Files**: Securely created and removed on error, but ensure the filesystem is secure.

## Limitations
- Files must be in the current directory (no paths or hidden files).
- Fixed 8 MiB chunk size (not configurable).
- No password-based encryption.
- No progress bar for large files.
- Limited permission checks on non-Unix systems (e.g., Windows).

## Building and Testing

### Build
```bash
cargo build --release
```

### Testing
Add tests with the `tempfile` crate. Add to `Cargo.toml`:
```toml
[dev-dependencies]
tempfile = "3.10"
```

Example test cases:
- Encrypt/decrypt empty files.
- Encrypt/decrypt small (<8 MiB) and large (>8 MiB) files.
- Test corrupted files and wrong keys.

Run tests:
```bash
cargo test
```

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit changes (`git commit -m "Add my feature"`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a pull request.

Ensure code follows Rust conventions and includes tests for new features.

## License
Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

## Contact
For issues or questions, open a GitHub issue or contact the maintainer at [your-email@example.com].