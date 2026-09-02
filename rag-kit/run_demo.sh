#!/bin/bash
# 知擎 RAG 控制台 · 一键启动脚本
# 用法： bash run_demo.sh   然后浏览器打开 http://localhost:8080
set -e
cd /Users/dagongjidecuoyiban/Desktop/workBuddy/2026-08-31-15-37-21/rag-freelance-kit/rag-kit
PORT=8080 /Users/dagongjidecuoyiban/anaconda3/envs/rag-forge/bin/python web_demo.py
