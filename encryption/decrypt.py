"""
Decryption Script for MedGraph Challenge (ORGANIZERS ONLY)

This script decrypts participant submissions using the private key.
It is called by GitHub Actions during automated evaluation.

⚠️  WARNING: This script requires the PRIVATE KEY which should NEVER
be committed to the repository. It is stored in GitHub Secrets.

Usage (in GitHub Actions):
    python decrypt.py <encrypted.enc> <private_key.pem> <output.csv>
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
    sys.exit(1)


def load_private_key(key_path: str, password: bytes = None):
    """Load RSA private key from PEM file."""
    with open(key_path, 'rb') as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=password,
            backend=default_backend()
        )
    return private_key


def decrypt_file(input_path: str, private_key_path: str, output_path: str):
    """
    Decrypt an encrypted submission file.
    
    Process:
    1. Load encrypted data structure
    2. Decrypt AES key using RSA private key
    3. Decrypt data using AES-256-GCM
    4. Save decrypted CSV
    """
    # Load encrypted data
    with open(input_path, 'r') as f:
        encrypted_data = json.load(f)
    
    # Validate structure
    required_fields = ['encrypted_key', 'nonce', 'ciphertext']
    for field in required_fields:
        if field not in encrypted_data:
            print(f"❌ Error: Missing field '{field}' in encrypted file")
            sys.exit(1)
    
    # Decode base64 fields
    encrypted_key = base64.b64decode(encrypted_data['encrypted_key'])
    nonce = base64.b64decode(encrypted_data['nonce'])
    ciphertext = base64.b64decode(encrypted_data['ciphertext'])
    
    # Load private key
    private_key = load_private_key(private_key_path)
    
    # Decrypt AES key with RSA
    try:
        aes_key = private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    except Exception as e:
        print(f"❌ Error decrypting AES key: {e}")
        sys.exit(1)
    
    # Decrypt data with AES-GCM
    try:
        aesgcm = AESGCM(aes_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as e:
        print(f"❌ Error decrypting data: {e}")
        sys.exit(1)
    
    # Save decrypted file
    with open(output_path, 'wb') as f:
        f.write(plaintext)
    
    print(f"✅ Decrypted file saved: {output_path}")
    
    return output_path


def main():
    if len(sys.argv) != 4:
        print("Usage: python decrypt.py <encrypted.enc> <private_key.pem> <output.csv>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    private_key_path = sys.argv[2]
    output_path = sys.argv[3]
    
    decrypt_file(input_path, private_key_path, output_path)


if __name__ == '__main__':
    main()
