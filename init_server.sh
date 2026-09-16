#!/bin/bash

set -e

echo "=========================================="
echo " DATAPULSE - Inicialização do servidor"
echo "=========================================="

# ------------------------------------------
# 1. Atualizar sistema
# ------------------------------------------

echo "[1/5] Atualizando sistema..."

sudo apt-get update
sudo apt-get upgrade -y


# ------------------------------------------
# 2. Instalar dependências
# ------------------------------------------

echo "[2/5] Instalando dependências..."

sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    git


# ------------------------------------------
# 3. Instalar Docker
# ------------------------------------------

echo "[3/5] Instalando Docker..."

# Criar diretório para chave GPG
sudo install -m 0755 -d /etc/apt/keyrings

# Adicionar chave oficial do Docker
sudo curl -fsSL \
    https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc

sudo chmod a+r /etc/apt/keyrings/docker.asc

# Adicionar repositório oficial do Docker
echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") \
  stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Atualizar repositórios
sudo apt-get update

# Instalar Docker Engine + Compose
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin


# ------------------------------------------
# 4. Habilitar Docker
# ------------------------------------------

echo "[4/5] Configurando Docker..."

sudo systemctl enable docker
sudo systemctl enable containerd

sudo systemctl start docker
sudo systemctl start containerd


# ------------------------------------------
# 5. Permitir Docker sem sudo
# ------------------------------------------

echo "[5/5] Configurando usuário..."

sudo usermod -aG docker "$USER"


# ------------------------------------------
# Verificações
# ------------------------------------------

echo ""
echo "=========================================="
echo " VERIFICAÇÃO"
echo "=========================================="

echo ""
echo "Git:"
git --version

echo ""
echo "Docker:"
docker --version

echo ""
echo "Docker Compose:"
docker compose version

echo ""
echo "Docker Status:"
sudo systemctl is-active docker

echo ""
echo "=========================================="
echo " INSTALAÇÃO CONCLUÍDA"
echo "=========================================="

echo ""
echo "IMPORTANTE:"
echo "Faça logout/login novamente para usar"
echo "Docker sem sudo."
echo ""
echo "Depois teste:"
echo "  docker run hello-world"
echo "  docker compose version"
echo ""