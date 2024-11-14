更新内容: 调整数据生成方式
===
feat
---
1. 训练配置适配practice库的resnet参数结构
2. 推理配置新增image_width和image_height参数，调整字体大小范围，适配practice库的resnet参数结构
3. 推理脚本适配Inference类的输出
4. 推理数据集inference.py提前调整图像大小，不在前处理调整
5. 虚拟数据集一次载入多个视频，忽略行数过少的文本
6. transforms新增随机网格掩膜
7. Inference类输出结果调整
8. Model类适配practice库的模型结构
9. Postprocessor类新增文本框坐标输出
10. 训练器每5个epoch重新load数据
---
fix
---
1. 真实数据集gt.py修复label文件和视频文件不匹配的问题