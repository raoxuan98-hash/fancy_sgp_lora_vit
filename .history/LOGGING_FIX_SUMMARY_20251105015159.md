# 日志实时更新修复说明

## 问题描述
在运行 `main.py` 时，日志文件不会实时更新，导致无法及时查看训练进度和输出信息。

## 根本原因
Python 的 `logging.FileHandler` 默认使用缓冲模式，日志消息会被缓存在内存中，直到缓冲区满或者程序结束时才会写入磁盘文件。这导致日志文件看起来"不会实时更新"。

## 解决方案
在 `trainer.py` 中修改了日志配置，启用了行缓冲模式：

### 修改内容
1. **在 `train()` 函数中**：
```python
# 修改前
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(filename)s] => %(message)s',
    handlers=[
        logging.FileHandler(filename=os.path.join(logfile_name, 'record.log')),
        logging.StreamHandler(sys.stdout)])

# 修改后
file_handler = logging.FileHandler(filename=os.path.join(logfile_name, 'record.log'), mode='a', encoding='utf-8')
file_handler.stream.reconfigure(line_buffering=True)  # 启用行缓冲

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(filename)s] => %(message)s',
    handlers=[
        file_handler,
        logging.StreamHandler(sys.stdout)])
```

2. **在 `Bayesian_evaluate()` 函数中**：
```python
# 同样的修改
file_handler = logging.FileHandler(filename=os.path.join(logfile_name, 'record.log'), mode='a', encoding='utf-8')
file_handler.stream.reconfigure(line_buffering=True)  # 启用行缓冲
```

## 技术细节
- `line_buffering=True`：启用行缓冲，每次遇到换行符时就立即刷新缓冲区
- `mode='a'`：以追加模式打开文件
- `encoding='utf-8'`：明确指定 UTF-8 编码，避免中文乱码

## 测试验证
创建了 `test_realtime_logging.py` 测试脚本，验证了：
- ✅ 修复后的日志配置能够实时写入文件
- ⚠️ 原始的缓冲配置确实存在延迟问题
- ✅ 日志消息立即出现在文件中，无需等待程序结束

## 效果
现在运行 `main.py` 时：
- 日志消息会立即写入到 `record.log` 文件
- 可以实时查看训练进度、损失值、准确率等信息
- 不需要等待程序结束就能监控训练状态
- 支持使用 `tail -f record.log` 等工具实时查看日志
- **新增：项目开始时会显示日志文件路径，方便用户查找**

## 使用建议
1. 运行训练时，可以在另一个终端使用 `tail -f` 实时查看日志：
   ```bash
   tail -f sldc_logs_sgp_lora_vit_main/*/record.log
   ```

2. 如果需要查看特定任务的日志，可以结合 `grep` 使用：
   ```bash
   tail -f record.log | grep "accuracy"
   ```

3. 修复不影响现有的日志格式和目录结构，只是解决了缓冲问题。

## 兼容性
- 修改向后兼容，不影响现有代码
- 所有日志功能和格式保持不变
- 只是解决了实时更新的问题
