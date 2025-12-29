# nohup python3 main.py "0" "1" &
# pid1=$!
# nohup python3 main.py "1" "2" &
# pid2=$!
# nohup python3 main.py "2" "3" &
# pid3=$!
# nohup python3 main.py "3" "4" &
# pid4=$!
# wait $pid1 $pid2 $pid3 $pid4
# python main.py "0" "1"

# nohup python main.py "0" "1" > log0.out 2>&1
# nohup python main.py "1" "2" > log1.out 2>&1
# nohup python main.py "2" "3" > log2.out 2>&1
# nohup python main.py "3" "4" > log3.out 2>&1


# set -e  # 只要有命令出错就退出
# set -u  # 变量未定义就报错
# set -o pipefail  # 管道中任何一环出错整个脚本都当失败

# echo "[INFO] Adjusting ulimit settings..."

# # 放宽ulimit，避免各种隐藏的资源限制
# ulimit -c unlimited       # 允许生成core文件
# ulimit -n 65535           # 文件句柄数限制
# ulimit -s 65535           # stack大小
# ulimit -l unlimited       # 锁定内存大小

# echo "[INFO] ulimit settings updated:"
# ulimit -a

# echo "[INFO] Setting core dump path..."

# # 设置core dump保存路径，可以根据需要修改路径
# CORE_DUMP_PATH="/var/crash/core.%e.%p"
# mkdir -p $(dirname $CORE_DUMP_PATH)  # 确保路径存在
# echo "/var/crash/core.%e.%p" > /proc/sys/kernel/core_pattern

# echo "[INFO] Core dump path set to $CORE_DUMP_PATH"

# echo "[INFO] Starting training..."

# 启动Python训练脚本
# nohup python3 main.py "0" "1" &
# pid1=$!
# nohup python3 main.py "1" "2" &
# pid2=$!
# nohup python3 main.py "2" "3" &
# pid3=$!
# nohup python3 main.py "3" "4" &
# pid4=$!
# wait $pid1 $pid2 $pid3 $pid4
python3 main.py "0" "1"
# echo "[INFO] Training completed successfully."

# 定期清理旧日志：  rm -rf logs/version_*   