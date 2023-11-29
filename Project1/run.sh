#!/bin/bash

# 循环遍历当前文件夹中的Python脚本
for script in *.py; do
    if [ -f "$script" ]; then
        echo "Running $script"
        python "$script"
        echo "$script executed."
    fi
done
