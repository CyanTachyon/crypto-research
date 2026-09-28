#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d "node_modules" ]; then
  npm install
fi

mkdir -p data

MODE=sim node --import tsx src/index.ts
