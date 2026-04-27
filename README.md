# Download pre-trained models from HuggingFace
uv run yale-qc download-models

# Download 5,000 images (seed 1337, max 1500px, 20 threads)
uv run yale-qc download -n 5000

# Augment for all QC categories
uv run yale-qc augment

# Prepare datasets for YOLO training
uv run yale-qc prepare

# Train all YOLO26 models
uv run yale-qc train

# Run the web app with green/red QC status lights
uv run yale-qc serve
uv run yale-qc serve --host 0.0.0.0

# On macOS, find your LAN IP:
ipconfig getifaddr en0    # WiFi
ipconfig getifaddr en1    # Ethernet

# Analyze a single image
uv run yale-qc analyze path/to/image.jpg


# ============================================================
# Running the QC server on a Windows scanning workstation
# ============================================================

A `start-qc.bat` launcher is included at the project root.

What it does:
  1. Detects the machine's LAN IPv4 address.
  2. Writes the access URL to `logs\server-url.txt` (always overwritten,
     so this file is the single source of truth for "where do I connect?").
  3. Creates a timestamped log file `logs\qc-YYYYMMDD_HHMMSS.log` capturing
     all stdout + stderr from the server.
  4. Launches `uv run yale-qc serve --host 0.0.0.0 --port 5000`.

Manual run (one-off):
  Double-click `start-qc.bat`, OR from cmd:
      cd C:\path\to\yale-scanning-qc
      start-qc.bat

  After it starts, open `logs\server-url.txt` for the URL, e.g.
  `http://192.168.1.42:5000`. Connect from any machine on the same network.


# ------------------------------------------------------------
# Run automatically when Windows boots (Task Scheduler)
# ------------------------------------------------------------

These steps make the QC server start before anyone logs in, so the
scanning workstation is ready as soon as it powers on.

  1. Press Win + R, type `taskschd.msc`, press Enter.
     (Task Scheduler opens.)

  2. In the right-hand pane click "Create Task..." (NOT "Create Basic Task" —
     we need the advanced options).

  3. **General** tab:
       - Name:        Yale Scanning QC
       - Description: Starts the QC web server at boot
       - Select "Run whether user is logged on or not"
       - Check       "Run with highest privileges"
       - Configure for: Windows 10 (or Windows 11)

  4. **Triggers** tab → New...
       - Begin the task: At startup
       - (Leave all advanced settings at defaults; click OK.)

  5. **Actions** tab → New...
       - Action:           Start a program
       - Program/script:   C:\path\to\yale-scanning-qc\start-qc.bat
       - Start in (optional): C:\path\to\yale-scanning-qc
         (IMPORTANT — without "Start in", `uv` won't find pyproject.toml.)
       - Click OK.

  6. **Conditions** tab:
       - Uncheck "Start the task only if the computer is on AC power"
         (otherwise it won't run on battery / unplugged moments).

  7. **Settings** tab:
       - Check "If the task fails, restart every: 1 minute" (3 attempts).
       - Set "Stop the task if it runs longer than:" → No (uncheck it).
       - Click OK. Windows will prompt for the user account password
         it should run as — use an account that has access to your
         scanning directories.

  8. Reboot the machine. After it boots:
       - Open `C:\path\to\yale-scanning-qc\logs\server-url.txt` to see
         the access URL.
       - From any machine on the same network, browse to that URL.
       - Sign in with the credentials in `.env` (default: admin /
         Voynichisabook).

Tip: If the scheduled task doesn't start the server, open the latest
`logs\qc-*.log` file — the full stdout + stderr is captured there.


# ============================================================
# Moving scans from DELL-i2S to the DSCA AI Project share
# ============================================================

`move-scans-to-share.bat` (at the project root) replaces Ja'nya's
manual end-of-shift move. Double-click it (or a desktop shortcut to
it) and it will:

  1. Verify the destination share is reachable. If not, it aborts
     BEFORE moving anything and prints a clear error.
  2. Move every image file (*.tif, *.tiff, *.jpg, *.jpeg, *.png, and
     common RAW types) from the configured source to the destination.
     Non-image files on the Desktop (shortcuts, .docx, etc.) are
     never touched.
  3. Use `robocopy /MOV` so files are only deleted from the source
     AFTER a verified copy at the destination.
  4. Log everything to `logs\move-YYYYMMDD_HHMMSS.log` and pause at
     the end so the operator can confirm the result.


# ------------------------------------------------------------
# One-time setup on DELL-i2S
# ------------------------------------------------------------

  1. Open `move-scans-to-share.bat` in Notepad and confirm the two
     CONFIG lines near the top:

       set "SOURCE=%USERPROFILE%\Desktop"
       set "DEST=Y:\AI Project"

     - SOURCE: leave as `%USERPROFILE%\Desktop` if Ja'nya saves
       scans loose on the Desktop. Change to e.g.
       `%USERPROFILE%\Desktop\Scans` if she puts them in a subfolder.
     - DEST: must be the WINDOWS path to the DSCA AI Project share.
       Either a mapped drive letter (`Y:\AI Project`) or the full
       UNC path (`\\fileserver.yale.edu\DSCA_Digital_Studio-CC1702-LIBITS\AI Project`).
       Open File Explorer → "This PC" to confirm which letter the
       share is mounted under.

  2. Make a Desktop shortcut so it's one click for Ja'nya:
       - Right-click `move-scans-to-share.bat` → "Create shortcut"
       - Drag the resulting shortcut onto the Desktop
       - (Optional) rename it to "Move scans to Y drive"
       - (Optional) right-click the shortcut → Properties → Change
         Icon... and pick a folder/arrow icon so it stands out.

  3. Test it once with a single throwaway image on the Desktop
     before relying on it for a full day's scans.

Tip: If something looks wrong, the latest `logs\move-*.log` has the
complete robocopy output (including any files that failed to copy
and were therefore NOT deleted from the source).
