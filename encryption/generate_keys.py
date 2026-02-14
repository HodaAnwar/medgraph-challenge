"""
Key Generation Script for MedGraph Challenge (ORGANIZERS ONLY)

This script generates RSA-2048 key pairs for the encryption system.
Run this ONCE to create keys, then:
1. Add public_key.pem to the repository (for participants)
2. Add private_key.pem to GitHub Secrets (NEVER commit!)

Usage:
    python generate_keys.py [output_directory]
    
Example:
    python generate_keys.py ./keys/
"""

import sys
import os
from pathlib import Path

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
except ImportError:
    print("Error: cryptography package not installed.")
    print("Please run: pip install cryptography")
    sys.exit(1)


def generate_key_pair(output_dir: str = '.'):
    """
    Generate RSA-2048 key pair for the competition.
    
    Creates:
    - public_key.pem: Share with participants (commit to repo)
    - private_key.pem: Keep SECRET (add to GitHub Secrets)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("MedGraph Challenge - Key Generation")
    print("="*60)
    
    # Generate private key
    print("\n🔐 Generating RSA-2048 key pair...")
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    # Get public key
    public_key = private_key.public_key()
    
    # Serialize private key
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Serialize public key
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    # Save keys
    private_key_path = output_path / 'private_key.pem'
    public_key_path = output_path / 'public_key.pem'
    
    with open(private_key_path, 'wb') as f:
        f.write(private_pem)
    
    with open(public_key_path, 'wb') as f:
        f.write(public_pem)
    
    print(f"\n✅ Keys generated successfully!")
    print(f"\n📁 Output files:")
    print(f"   Public key:  {public_key_path}")
    print(f"   Private key: {private_key_path}")
    
    print("\n" + "="*60)
    print("⚠️  IMPORTANT SECURITY INSTRUCTIONS")
    print("="*60)
    
    print("""
1. PUBLIC KEY (public_key.pem):
   ✅ Commit to repository in encryption/ folder
   ✅ Participants will use this to encrypt submissions
   
2. PRIVATE KEY (private_key.pem):
   ❌ NEVER commit to the repository
   ❌ NEVER share with anyone
   ✅ Add to GitHub Secrets as 'PRIVATE_KEY'
   
To add to GitHub Secrets:
   1. Go to your repository Settings
   2. Click "Secrets and variables" → "Actions"
   3. Click "New repository secret"
   4. Name: PRIVATE_KEY
   5. Value: Copy entire contents of private_key.pem
   6. Click "Add secret"
   
After adding to GitHub Secrets, DELETE the local private_key.pem file!
""")
    
    # Print the public key for reference
    print("\n📋 Public Key (for reference):")
    print("-"*60)
    print(public_pem.decode('utf-8'))
    
    return public_key_path, private_key_path


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    generate_key_pair(output_dir)


if __name__ == '__main__':
    main()
