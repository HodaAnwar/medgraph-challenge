"""
Encryption Script for MedGraph Challenge Submissions

This script encrypts your prediction CSV file using RSA + AES hybrid encryption.
The encrypted file can ONLY be decrypted by the competition organizers using
their private key (stored securely in GitHub Secrets).

Usage:
    python encrypt.py <predictions.csv> <public_key.pem> <output.enc>
    
Example:
    python encrypt.py my_predictions.csv ../encryption/public_key.pem ../submissions/my_team.enc

Security:
    - Uses RSA-2048 for key encryption
    - Uses AES-256-GCM for data encryption
    - Your predictions are completely unreadable without the private key
"""

import sys
import os
import json
import base64
from pathlib import Path

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.backends import default_backend
except ImportError:
    print("Error: cryptography package not installed.")
    print("Please run: pip install cryptography")
    sys.exit(1)


def load_public_key(key_path: str):
    """Load RSA public key from PEM file."""
    with open(key_path, 'rb') as f:
        public_key = serialization.load_pem_public_key(
            f.read(),
            backend=default_backend()
        )
    return public_key


def encrypt_file(input_path: str, public_key_path: str, output_path: str):
    """
    Encrypt a CSV file using hybrid RSA + AES encryption.
    
    Process:
    1. Generate random AES-256 key
    2. Encrypt the CSV data with AES-256-GCM
    3. Encrypt the AES key with RSA public key
    4. Save both to output file
    """
    # Validate input file
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if not input_path.endswith('.csv'):
        print(f"⚠️  Warning: Input file should be a CSV file")
    
    # Read input data
    with open(input_path, 'rb') as f:
        plaintext = f.read()
    
    print(f"📄 Input file: {input_path}")
    print(f"   Size: {len(plaintext)} bytes")
    
    # Validate CSV format
    try:
        content = plaintext.decode('utf-8')
        lines = content.strip().split('\n')
        if len(lines) < 2:
            print("❌ Error: CSV file appears to be empty")
            sys.exit(1)
        
        header = lines[0].lower()
        if 'graph_id' not in header or 'prediction' not in header:
            print("❌ Error: CSV must have 'graph_id' and 'prediction' columns")
            sys.exit(1)
        
        print(f"   Predictions: {len(lines) - 1} rows")
    except Exception as e:
        print(f"❌ Error validating CSV: {e}")
        sys.exit(1)
    
    # Load public key
    print(f"\n🔑 Loading public key: {public_key_path}")
    public_key = load_public_key(public_key_path)
    
    # Generate random AES key (256 bits)
    aes_key = os.urandom(32)
    
    # Generate random nonce for AES-GCM (96 bits)
    nonce = os.urandom(12)
    
    # Encrypt data with AES-256-GCM
    print("\n🔐 Encrypting predictions with AES-256-GCM...")
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    
    # Encrypt AES key with RSA public key
    print("🔐 Encrypting AES key with RSA-2048...")
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Create output structure
    output_data = {
        'version': '1.0',
        'algorithm': 'RSA-2048-OAEP-SHA256 + AES-256-GCM',
        'encrypted_key': base64.b64encode(encrypted_key).decode('utf-8'),
        'nonce': base64.b64encode(nonce).decode('utf-8'),
        'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
    }
    
    # Save encrypted file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Encrypted file saved: {output_path}")
    print(f"   Size: {os.path.getsize(output_path)} bytes")
    
    print("\n" + "="*60)
    print("🎉 ENCRYPTION COMPLETE!")
    print("="*60)
    print(f"\nYour encrypted submission: {output_path}")
    print("\nNext steps:")
    print("1. Fork the repository (if not already)")
    print("2. Add your .enc file to the submissions/ folder")
    print("3. Create a Pull Request")
    print("4. Wait for automated evaluation (2-5 minutes)")
    print("\n⚠️  REMINDER: You can only submit ONCE!")
    
    return output_path


def main():
    if len(sys.argv) != 4:
        print("Usage: python encrypt.py <predictions.csv> <public_key.pem> <output.enc>")
        print("\nExample:")
        print("  python encrypt.py my_predictions.csv encryption/public_key.pem submissions/my_team.enc")
        sys.exit(1)
    
    input_path = sys.argv[1]
    public_key_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Ensure output has .enc extension
    if not output_path.endswith('.enc'):
        output_path += '.enc'
    
    encrypt_file(input_path, public_key_path, output_path)


if __name__ == '__main__':
    main()
