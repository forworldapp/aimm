# ☁️ Deploying GRVT Bot to Oracle Cloud (Free Tier)

This guide documents the step-by-step process to deploy the GRVT Market Maker Bot to **Oracle Cloud Infrastructure (OCI)** using the **Always Free** tier in the **Tokyo Region**.

The Tokyo region is recommended for its low latency connection to GRVT exchange servers (hosted in AWS Tokyo).

---

## 📋 1. Prerequisites & Account Setup

1.  **Sign Up**: Go to [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/) and click "Start for free".
2.  **Region Selection**:
    *   **Crucial**: Select **Japan East (Tokyo)** as your "Home Region". You cannot change this later easily for Free Tier resources.
    *   *Note*: If Tokyo is unavailable/full, Seoul is a decent alternative, but Tokyo is best for latency (~1-2ms ping to AWS Tokyo).
3.  **Verification**: You will need a credit card for identity verification (small temporary charge, refunded).

---

## 🖥️ 2. Create Compute Instance (Server)

Once logged into the OCI Console:

1.  Navigate to **Compute** -> **Instances**.
2.  Click **Create Instance**.
3.  **Name**: e.g., `grvt-bot-server`.
4.  **Image & Shape** (The most important part):
    *   Click **Change Image**. Select **Canonical Ubuntu 22.04** or **24.04** (Minimal or Standard).
    *   Click **Change Shape**. Select **Ampere** -> **VM.Standard.A1.Flex**.
    *   **Config**: Drag the sliders to **4 OCPUs** and **24 GB RAM**. (This is free!)
    *   *Note*: If you see "Out of capacity", retry later or choose `VM.Standard.E2.1.Micro` (AMD, much slower, 1GB RAM) as a temporary fallback.
5.  **Networking**:
    *   Create new VCN (Virtual Cloud Network).
    *   Assign a **Public IP address**.
6.  **SSH Keys**:
    *   Select **"Generate a key pair for me"**.
    *   **DOWNLOAD the Private Key (.key)** and Public Key.
    *   *Keep the Private Key safe! You cannot access the server without it.*
7.  Click **Create**. Wait for the instance to turn "Running".

---

## 🛡️ 3. Network & Security Settings

We need to open ports for **SSH (22)** (default open) and **Streamlit Dashboard (8501)**.

1.  **OCI VCN Security List**:
    *   Go to **Networking** -> **Virtual Cloud Networks**.
    *   Click on your VCN -> **Subnets** -> Click the public subnet.
    *   Click **Security Lists** -> **Default Security List**.
    *   **Add Ingress Rules**:
        *   Source CIDR: `0.0.0.0/0` (Access from anywhere) or Your Home IP.
        *   Protocol: TCP
        *   Destination Port Range: `8501`
        *   Description: Streamlit Dashboard
2.  **OS Firewall (Ubuntu)**:
    *   You will configure this inside the server in the next step.

---

## 🔧 4. Server Setup & Installation

### 4.1 Connect to Server

Open your terminal (PowerShell or Git Bash on Windows):
```bash
# Move key to a safe folder and set permissions (Linux/Mac only, Windows users just use path)
ssh -i "path/to/ssh-key-2024-xx-xx.key" ubuntu@<PUBLIC-IP-ADDRESS>
```

### 4.2 System Update & Tools

Once connected:
```bash
# Update System
sudo apt update && sudo apt upgrade -y

# Install Python & Git & Pip
sudo apt install python3-pip python3-venv git htop screen -y

# Allow Port 8501 in OS Firewall
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8501 -j ACCEPT
sudo netfilter-persistent save
```

### 4.3 Clone Repository & Install Bot

```bash
# Clone your repo (Assuming you pushed it to GitHub/GitLab)
git clone <YOUR_GIT_REPO_URL> grvt_bot
cd grvt_bot

# Setup Virtual Environment (Recommended)
python3 -m venv venv
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### 4.4 Configure Bot

1.  Create `.env` file:
    ```bash
    nano .env
    ```
    *   Paste your `GRVT_API_KEY` and `GRVT_PRIVATE_KEY` here.
    *   Save: `Ctrl+O`, `Enter`, `Ctrl+X`.

2.  Review `config.yaml` if needed (e.g., enable Live Mode).

---

## 🤖 5. Run & Keep Alive (24/7)

If you just run `python main.py`, the bot dies when you close SSH. We use `screen` or `systemd`.

### Method A: Using `screen` (Simple)

```bash
# Create a new session named 'bot'
screen -S bot

# Activate Environment
source venv/bin/activate

# Run Streamlit (Dashboard) in background & Bot
# We can run both commands
streamlit run dashboard.py --server.port 8501 --server.Headless true &
python main.py
```
*   **Detach**: Press `Ctrl+A`, then `D`. (Bot keeps running).
*   **Reattach**: `screen -r bot`.

### Method B: Using `systemd` (Professional, Auto-restart)

Create a service file: `sudo nano /etc/systemd/system/grvt-bot.service`

```ini
[Unit]
Description=GRVT Market Maker Bot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/grvt_bot
ExecStart=/home/ubuntu/grvt_bot/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable grvt-bot
sudo systemctl start grvt-bot
```
*(Repeat for Streamlit if needed, or run Streamlit in screen)*

---

## 📊 6. Access Dashboard

Open your browser and go to:
`http://<SERVER-PUBLIC-IP>:8501`

You should see your bot running with low latency!
