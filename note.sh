sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean

sudo apt update
sudo apt install -f

ubuntu-drivers devices
# sudo ubuntu-drivers autoinstall
sudo apt update
sudo apt install nvidia-driver-535 nvidia-dkms-535 nvidia-utils-535
sudo reboot
nvidia-smi