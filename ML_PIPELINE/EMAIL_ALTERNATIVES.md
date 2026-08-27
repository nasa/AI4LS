# Alternative Email Authentication Methods

If your Google account doesn't support app passwords, here are several alternatives:

## Option 1: Use Environment Variables (Most Secure) ⭐ RECOMMENDED

Store credentials in environment variables instead of typing them:

```bash
# Set environment variables (only in current terminal session)
export EMAIL_USER=your.email@gmail.com
export EMAIL_PASSWORD=your_password

# Run pipeline (no prompts!)
python run_pipeline_background_v2.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email results@example.com
```

**Benefits:**
- Password not visible in command history
- Password not visible in shell
- Can be stored in `.bashrc` or `.zshrc` for reuse
- Works with any email provider

**To make it persistent (add to ~/.bashrc or ~/.zshrc):**

```bash
# Add these lines to ~/.bashrc or ~/.zshrc
export EMAIL_USER="your.email@gmail.com"
export EMAIL_PASSWORD="your_password"
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

---

## Option 2: Use Outlook/Office 365

If you have an Outlook account:

```bash
python run_pipeline_background_v2.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email results@example.com \
    --provider outlook
```

When prompted:
- Email: your@outlook.com
- Password: your regular Outlook password

---

## Option 3: Use Yahoo Mail

If you have a Yahoo account:

```bash
python run_pipeline_background_v2.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email results@example.com \
    --provider yahoo
```

Note: Yahoo may require an [app password](https://login.yahoo.com/account/security)

---

## Option 4: Use SendGrid (Free Tier Available)

If you want a transactional email service (free):

1. Sign up at https://sendgrid.com (free tier available)
2. Create an API key
3. Use it like this:

```bash
python run_pipeline_background_v2.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email results@example.com \
    --provider sendgrid \
    --sender-email apikey \
    --sender-password "SG.your_actual_api_key_here"
```

**Or with environment variables:**

```bash
export EMAIL_USER="apikey"
export EMAIL_PASSWORD="SG.your_api_key"

python run_pipeline_background_v2.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email results@example.com \
    --provider sendgrid
```

---

## Option 5: Use Custom SMTP Server

If your organization has an internal email server:

```bash
python run_pipeline_background_v2.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email results@example.com \
    --smtp-server smtp.yourcompany.com \
    --smtp-port 587
```

When prompted:
- Email: your.email@company.com
- Password: your company email password

---

## Option 6: Use Gmail Regular Password (Risky)

**⚠️ NOT RECOMMENDED** - Only if your account doesn't support app passwords:

1. Go to https://myaccount.google.com/security
2. Enable "Less secure app access" (⚠️ Security risk!)
3. Run:

```bash
export EMAIL_USER="your.email@gmail.com"
export EMAIL_PASSWORD="your_gmail_password"

python run_pipeline_background_v2.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email results@example.com
```

This is less secure because the app gets full access to your Gmail account.

---

## Recommended Setup

**Best practice: Environment variables + your preferred provider**

### For Gmail:
```bash
# Try to get app password working (recommended)
# If not, fall back to environment variables

export EMAIL_USER="your.email@gmail.com"
export EMAIL_PASSWORD="your_app_password"

python run_pipeline_background_v2.py --tissue liver -tc "Factor Value[Spaceflight]" --email results@example.com
```

### For Outlook:
```bash
export EMAIL_USER="your@outlook.com"
export EMAIL_PASSWORD="your_outlook_password"

python run_pipeline_background_v2.py --tissue liver -tc "Factor Value[Spaceflight]" --email results@example.com --provider outlook
```

### For SendGrid:
```bash
export EMAIL_USER="apikey"
export EMAIL_PASSWORD="SG.your_api_key"

python run_pipeline_background_v2.py --tissue liver -tc "Factor Value[Spaceflight]" --email results@example.com --provider sendgrid
```

---

## Supported Email Providers

```
Gmail            (SMTP: smtp.gmail.com:587)
Outlook/O365     (SMTP: smtp.office365.com:587)
Yahoo            (SMTP: smtp.mail.yahoo.com:587)
SendGrid         (SMTP: smtp.sendgrid.net:587)
Mailgun          (SMTP: smtp.mailgun.org:587)
Custom Server    (Use --smtp-server and --smtp-port)
```

---

## Troubleshooting

### Error: "Authentication failed"
- Check username and password are correct
- For Gmail: Make sure you're using app password, not regular password
- For SendGrid: Username must be `apikey` exactly

### Error: "Connection refused"
- Check SMTP server and port are correct
- Verify no firewall is blocking port 587

### Error: "Connection timeout"
- Try a different port: 465 (SSL) instead of 587 (TLS)
- Some providers use different ports

### Password leaked in history?
Use environment variables instead:
```bash
export EMAIL_PASSWORD="your_password"
# Now it's not in command history
```

---

## Creating a Config File (Advanced)

Create `email_config.json`:

```json
{
    "sender_email": "your.email@gmail.com",
    "sender_password": "your_app_password",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "use_tls": true
}
```

Then modify the script to read from this file instead of prompting.

---

Try **Option 1 (Environment Variables)** with your email provider first - it's the most secure and works with everything! 🎯
