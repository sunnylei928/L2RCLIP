import os
import re
import torch
torch.set_float32_matmul_precision('medium') # 开启 L40 显卡的狂暴加速模式
# 定义基础路径
results_dir = "/home/ubuntu/lq/L2RCLIP_results" # 建议换个独立文件夹存放 L2RCLIP 的结果
kfold_dir = "/home/ubuntu/lq/OrdinalCLIP/data/data_list/kfold"

# 1. 读取我们刚刚写好的 L2RCLIP 配置文件作为模板
with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    original_yaml_text = f.read()

# 真正执行 5 折交叉验证 (0 到 4)
folds_to_run = [0]

for i in folds_to_run:
    print(f"\n===================================================")
    print(f"                 🚀 开始训练 Fold {i}                ")
    print(f"===================================================")

    fold_result_path = os.path.join(results_dir, f"fold_{i}")

    # 2. 正则替换大法
    new_yaml_text = re.sub(r"output_dir:\s*.*", f"output_dir: {fold_result_path}", original_yaml_text)
    
    # 强制改写数据路径：训练集和验证集跟着循环 i 动态变化
    new_yaml_text = re.sub(r"train_data_file:\s*.*", f"train_data_file: {kfold_dir}/fold_{i}_train.txt", new_yaml_text)
    new_yaml_text = re.sub(r"val_data_file:\s*.*", f"val_data_file: {kfold_dir}/fold_{i}_val.txt", new_yaml_text)
    
    # 【核心修改点】测试集必须固定为独立文件！
    new_yaml_text = re.sub(r"test_data_file:\s*.*", f"test_data_file: {kfold_dir}/independent_test.txt", new_yaml_text)

    # 3. 另存为专门给这一折用的临时配置文件
    temp_config_path = f"configs/temp_l2rclip_fold_{i}.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        f.write(new_yaml_text)

    # 4. 调用命令行运行训练
    run_command = f"python run.py --config {temp_config_path}"
    print(f"执行命令: {run_command}\n")
    
    # 获取运行返回状态码 (0 表示正常结束)
    ret_code = os.system(run_command)

    # 5. 检查是否成功
    if ret_code == 0:
        print(f"✅ Fold {i} 训练完毕！结果已保存在 {fold_result_path}")
    else:
        print(f"❌ 致命错误：Fold {i} 运行报错退出了 (错误码 {ret_code})！")
        break # 遇到真实代码报错时立刻停下，方便排查
        
print("\n🎉 5折交叉验证全部脚本执行结束！")