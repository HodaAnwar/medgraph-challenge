# 🔐 Secure Submission System

This competition uses an **encrypted submission system** to keep test labels completely hidden from participants while enabling fully automated evaluation.

## 🛡️ Security Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PARTICIPANT (Public)                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Create predictions.csv                                       │
│  2. Encrypt with PUBLIC key → submission.enc                     │
│  3. Submit encrypted file via Pull Request                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GITHUB ACTIONS (Secret)                       │
├─────────────────────────────────────────────────────────────────┤
│  4. Decrypt with PRIVATE key (from GitHub Secrets)              │
│  5. Evaluate against hidden test labels                          │
│  6. Update leaderboard automatically                             │
│  7. Comment results on PR                                        │
└─────────────────────────────────────────────────────────────────┘
```

## 🔑 How It Works

### Encryption Algorithm
- **RSA-2048-OAEP-SHA256** for key encryption
- **AES-256-GCM** for data encryption
- Industry-standard hybrid encryption ensuring your predictions are unreadable without the private key

### Security Guarantees
- ✅ Test labels remain completely hidden
- ✅ Private key never exposed in logs or repository
- ✅ Encrypted submissions are unreadable by anyone
- ✅ Automated evaluation with no manual intervention
- ✅ One submission per participant enforced

---

## 📋 Submission Instructions

### Step 1: Install Dependencies

```bash
pip install cryptography pandas
```

### Step 2: Prepare Your Predictions

Create a CSV file with your predictions:

```csv
graph_id,prediction
test_0001,0
test_0002,2
test_0003,1
...
```

Where:
- `0` = Normal
- `1` = Benign  
- `2` = Malignant

### Step 3: Encrypt Your Submission

```bash
python encryption/encrypt.py your_predictions.csv encryption/public_key.pem submissions/your_team_name.enc
```

**Output:**
```
📄 Input file: your_predictions.csv
   Size: 12543 bytes
   Predictions: 500 rows

🔑 Loading public key: encryption/public_key.pem

🔐 Encrypting predictions with AES-256-GCM...
🔐 Encrypting AES key with RSA-2048...

✅ Encrypted file saved: submissions/your_team_name.enc
```

### Step 4: Submit via Pull Request

1. **Fork** this repository (if not already done)
2. **Add** your `.enc` file to the `submissions/` folder
3. **Commit** and **push** to your fork
4. **Create a Pull Request** to the main repository

### Step 5: Wait for Results

- GitHub Actions will automatically decrypt and evaluate your submission
- Results will be posted as a comment on your PR (2-5 minutes)
- Leaderboard updates in real-time

---

## ⚠️ Important Rules

### One Submission Only
Each participant is allowed **exactly ONE submission**. This is strictly enforced:
- The system checks for existing submissions before evaluation
- Duplicate submissions will be automatically rejected
- Make sure your submission is final before submitting

### Submission Format
- File must be named: `your_team_name.enc`
- Must contain predictions for all test graphs
- Predictions must be integers: 0, 1, or 2

### Naming Convention
Your team name becomes your identifier on the leaderboard. Choose wisely!

---

## 🔍 Troubleshooting

### "Decryption failed"
- Ensure you used the correct `public_key.pem` from this repository
- Do not modify the encrypted file after creation
- Re-encrypt if needed

### "Validation failed"
- Check your CSV format (must have `graph_id` and `prediction` columns)
- Ensure all test graph IDs are present
- Predictions must be 0, 1, or 2

### "Duplicate submission detected"
- You have already submitted once
- Contact organizers if you believe this is an error

---

## 📁 Files in This Directory

| File | Description |
|------|-------------|
| `encrypt.py` | Encryption script for participants |
| `decrypt.py` | Decryption script (used by GitHub Actions) |
| `generate_keys.py` | Key generation (organizers only) |
| `public_key.pem` | Public key for encryption (use this!) |

---

## 🔒 For Organizers

### Setting Up the Private Key

1. Generate keys (if not already done):
   ```bash
   python encryption/generate_keys.py
   ```

2. Add private key to GitHub Secrets:
   - Go to repository **Settings** → **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Name: `PRIVATE_KEY`
   - Value: Contents of `private_key.pem`
   - Click **Add secret**

3. **Delete** the local `private_key.pem` file

### Keeping Test Labels Hidden

- Store `test_labels.csv` in the repository (it's only used by GitHub Actions)
- The private key in GitHub Secrets ensures only the CI can decrypt submissions
- Never expose the private key in any logs or outputs
