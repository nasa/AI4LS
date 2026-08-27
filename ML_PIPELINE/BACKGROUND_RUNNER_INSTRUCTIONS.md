# Background Pipeline Runner with Email Notification

Run your pipeline in the background and get an email when it's done!

## Setup

### Step 1: Get Gmail App Password

You need a Gmail app password (not your regular Gmail password):

1. Go to: https://myaccount.google.com/apppasswords
2. Select "Mail" and "Windows Computer" (or your device)
3. Google will generate a 16-character password
4. **Copy it** - you'll need it below

### Step 2: Run Pipeline in Background

**Option A: Prompt for credentials (most secure)**

```bash
python run_pipeline_background.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email your.email@gmail.com
```

It will ask you for:
- Gmail address to send from
- Gmail app password (the 16-character one you generated above)

**Option B: Provide credentials (less secure, for automation)**

```bash
python run_pipeline_background.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email your.email@gmail.com \
    --sender-email your.gmail@gmail.com \
    --sender-password "your-16-char-app-password"
```

## Usage Examples

### Simple (tissue + target column)

```bash
python run_pipeline_background.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email results@example.com
```

### With more options

```bash
python run_pipeline_background.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email results@example.com \
    --algorithm random_forest \
    --min-features 500 \
    --test-size 0.1 \
    --no-kegg
```

### Explicit OSD IDs

```bash
python run_pipeline_background.py \
    --osd-ids 47,48,137 \
    -tc "Factor Value[Spaceflight]" \
    --email results@example.com
```

## What You Get

### Success Email
Contains:
- ✓ Confirmation that pipeline completed
- Timestamp
- Pipeline arguments used
- Path to results

### Failure Email
Contains:
- ✗ Error message
- Pipeline arguments
- Last 20 lines of log file for debugging
- Timestamp

## Monitoring

The pipeline runs in the background while logging to:
```
pipeline_background.log
```

You can watch it with:
```bash
tail -f pipeline_background.log
```

## Key Features

✅ **Background execution** - Run in background, terminal stays free
✅ **Email notification** - Get results via email when done
✅ **Error handling** - Failures are emailed with debug info
✅ **Secure** - Prompts for credentials (not visible in command)
✅ **Logging** - Full log file saved for debugging

## Arguments

```
Required:
  --tissue TISSUE                 Tissue type (e.g., liver)
  OR --osd-ids IDS               Comma-separated OSD IDs
  -tc, --target-column COL       Target column name
  --email EMAIL                  Email to send results to

Optional:
  --algorithm ALG                ML algorithm (default: random_forest)
  --min-features N               Min features after filtering (default: 1000)
  --test-size FRAC              Test set fraction (default: 0.2)
  --no-ensemble                 Skip ensemble training
  --no-kegg                     Skip KEGG analysis
  --no-feature-importance       Skip feature importance
  --sender-email EMAIL          Gmail to send from (prompted if not given)
  --sender-password PASS        Gmail app password (prompted if not given)
```

## Gmail Troubleshooting

### "Login failed" error
- Make sure you're using an **app password** (16 chars), not your Gmail password
- Generate one at: https://myaccount.google.com/apppasswords
- Make sure 2-factor authentication is enabled on your account

### "SMTPAuthenticationError"
- Your sender email and password don't match
- Try prompting for them instead of passing on command line

### "SMTPException: SMTP AUTH extension not supported"
- Your SMTP server might not support TLS
- Try changing `smtp_server` in the code to a different provider

## Running on a Remote Server

If running on a cloud server or remote machine, you can use `nohup`:

```bash
nohup python run_pipeline_background.py \
    --tissue liver \
    -tc "Factor Value[Spaceflight]" \
    --email results@example.com \
    > pipeline.out 2>&1 &
```

Then:
- `tail -f pipeline.out` to watch output
- Pipeline runs even if you disconnect
- Email sent when complete

## Advanced: Using System Scheduler

### macOS (launchd)

Create `~/Library/LaunchAgents/com.pipeline.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTDs PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pipeline.runner</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>/path/to/run_pipeline_background.py</string>
        <string>--tissue</string>
        <string>liver</string>
        <string>-tc</string>
        <string>Factor Value[Spaceflight]</string>
        <string>--email</string>
        <string>results@example.com</string>
    </array>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.pipeline.plist
```

### Linux (cron)

```bash
# Edit crontab
crontab -e

# Add a scheduled job (runs daily at 2 AM):
0 2 * * * cd /path/to/pipeline && python run_pipeline_background.py --tissue liver -tc "Factor Value[Spaceflight]" --email results@example.com
```

---

Enjoy hands-free pipeline runs! 🚀
