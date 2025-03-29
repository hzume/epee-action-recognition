curl -sS https://starship.rs/install.sh | sh -s -- --yes
echo 'eval "$(starship init bash)"' >> ~/.bashrc
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

. .venv/bin/activate

git config --global user.email "yota040@gmail.com"
git config --global user.name "hzume"
