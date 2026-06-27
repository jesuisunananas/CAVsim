#!/bin/bash

echo "Installing CAVsim Systemd Service..."

# Ensure we are running as root
if [ "$EUID" -ne 0 ]
  then echo "Please run as root (use sudo ./setup_service.sh)"
  exit
fi

# Copy the service file to the systemd directory
cp cavsim.service /etc/systemd/system/cavsim.service

# Reload systemd to recognize the new file
systemctl daemon-reload

# Enable the service so it starts on boot
systemctl enable cavsim.service

# Start the service right now
systemctl start cavsim.service

echo "Done! The pipeline is now running in the background."
echo "----------------------------------------------------"
echo "To check the status: systemctl status cavsim"
echo "To view live logs: journalctl -u cavsim -f"
echo "To stop the script: systemctl stop cavsim"
echo "To restart the script: systemctl restart cavsim"
