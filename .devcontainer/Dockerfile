# Pincher Dev Environment
# Based on the official Rust devcontainer image
FROM mcr.microsoft.com/devcontainers/rust:1-bullseye

# Install additional system packages needed for Pincher development
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libssl-dev \
        cmake \
        curl \
        git \
        ssh \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install cargo-watch for development hot-reload
RUN cargo install cargo-watch --locked

# Create the I2I vessel socket directory
RUN mkdir -p /tmp/i2i-vessel

# Set up the workspace
WORKDIR /workspaces/pincher
