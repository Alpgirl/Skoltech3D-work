#!/usr/bin/env bash

echo 'This script will reconnect ALL usb devices. Do you want to proceed? Type YES or anything else.'
read resp
if [ "$resp" != "YES" ]; then exit 1; fi


printf '\nUnbind Bus 02\n'
echo -n "0000:00:14.0" | tee /sys/bus/pci/drivers/xhci_hcd/unbind
sleep 1
printf '\nRebind Bus 02\n'
echo -n "0000:00:14.0" | tee /sys/bus/pci/drivers/xhci_hcd/bind

printf '\nUnbind Bus 06\n'
echo -n "0000:05:00.0" | tee /sys/bus/pci/drivers/xhci_hcd/unbind
sleep 1
printf '\nRebind Bus 06\n'
echo -n "0000:05:00.0" | tee /sys/bus/pci/drivers/xhci_hcd/bind
