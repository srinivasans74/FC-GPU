#!/bin/bash

CURRENT_VALUE=$(cat /sys/module/amdgpu/parameters/hws_max_conc_proc)
echo "Current hws_max_conc_proc value: $CURRENT_VALUE"

# If already set, exit early
if [ "$CURRENT_VALUE" -eq 1 ]; then
    echo "Already set to 1. No action needed."
    exit 0
fi

echo "This will attempt to unload the AMDGPU driver (any active graphics session may crash)."
echo "Setting hws_max_conc_proc=1..."
read -p "Press Enter to continue or Ctrl+C to cancel."

echo "Unloading amdgpu..."
if sudo modprobe -r amdgpu; then
    echo "Unloaded successfully."

    # Try to reload with new param
    echo "Reloading amdgpu with hws_max_conc_proc=1..."
    if sudo modprobe amdgpu hws_max_conc_proc=1; then
        echo "Module reloaded."
    else
        echo "Failed to reload amdgpu module."
        exit 1
    fi

    NEW_VALUE=$(cat /sys/module/amdgpu/parameters/hws_max_conc_proc)
    echo "New hws_max_conc_proc value: $NEW_VALUE"

    if [ "$NEW_VALUE" -eq 1 ]; then
        echo " Successfully updated to single-process time-slicing mode."
    else
        echo " Setting did not persist. Consider setting it permanently (see below)."
    fi

else
    echo "Failed to unload amdgpu. It's probably in use."
    echo
    echo "To set this permanently (recommended):"
    echo "1. Create or edit /etc/modprobe.d/amdgpu-options.conf"
    echo "   sudo nano /etc/modprobe.d/amdgpu-options.conf"
    echo "   Add this line:"
    echo "     options amdgpu hws_max_conc_proc=1"
    echo
    echo "2. Update initramfs:"
    echo "   For Ubuntu/Debian: sudo update-initramfs -u"
    echo "   For Fedora/RHEL:   sudo dracut -f"
    echo
    echo "3. Reboot your system:"
    echo "   sudo reboot"
fi