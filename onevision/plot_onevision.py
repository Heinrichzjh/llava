import json
import matplotlib.pyplot as plt

# === 修改这里：你的输出路径 ===
output_dir = "output_drive_action_onevision"
json_path = f"{output_dir}/trainer_state.json"

try:
    # 读取训练记录
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    history = data['log_history']
    
    # 提取 step 和 loss
    steps = []
    losses = []
    
    for entry in history:
        if 'loss' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
            
    if not steps:
        print("未找到 Loss 数据，请检查 json 文件内容。")
    else:
        # 画图
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label='Training Loss', color='#1f77b4', linewidth=2)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图片
        save_path = "training_loss_curve.png"
        plt.savefig(save_path, dpi=300)
        print(f"✅ 成功！图片已保存为: {save_path}")
        print("请在 VS Code 左侧文件列表点击该图片查看。")

except FileNotFoundError:
    print(f"❌ 找不到文件: {json_path}")
    print("请确认你的 output_model_user_lora_show 文件夹里有 trainer_state.json")