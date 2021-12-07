'''
Description: 
Version: 
Author: Leidi
Date: 2021-03-26 17:46:16
LastEditors: Leidi
LastEditTime: 2021-12-07 14:57:45
'''
import onnx
import torch
import os
import numpy as np
import netron
import onnxruntime as ort
from onnxsim import simplify

import models.backbones


network = r'iresnet50'
pth_path = r'weight/glint360k/backbone.pth'
export_onnx_file = r'onnx_output/arcface_backbone_glint360k.onnx'
input_shape = (3, 112, 112)  # 输入数据,改成自己的输入shape
batch_size = 1  # 批处理大小

# 基础设置
if not os.path.isdir('./onnx_output'):
    os.mkdir('./onnx_output')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = eval("models.backbones.{}".format(network))(False).cuda() # 选择模型结构
weight = torch.load(pth_path) # 读取权重文件
model.load_state_dict(weight) # 将权重部署至网络
model.eval()  # 将模型设定为评价模式

x = torch.randn(batch_size, *input_shape)   # 生成随机张量
x = x.to(device)

torch_output_value = model(x).cpu().detach().numpy()

torch.onnx.export(model,
                  x,
                  export_onnx_file,
                  opset_version=11,
                  export_params=True,
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=["input"],  # 输入名
                  output_names=["output"],  # 输出名
                  )

# onnxsimplify优化onnx网络图结构
onnx_model = onnx.load(export_onnx_file)  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, export_onnx_file)
print('finished exporting onnx')

# 检查输出onnx文件
test = onnx.load(export_onnx_file)
onnx.checker.check_model(test)

# 输出onnx的计算图
# print(onnx.helper.printable_graph(test.graph))
print("\nonnx output ==> Passed!\n")

# 计算转换后的输出误差
onnx_model = onnx.load_model(export_onnx_file)  # 读取onnx模型参数，构建模型
sess = ort.InferenceSession(onnx_model.SerializeToString()) # 推理模型成员初始化
sess.set_providers(['CPUExecutionProvider'])  # 将模型部署至cpu
input_name = sess.get_inputs()[0].name  # 读取网络输入名称
output_name = sess.get_outputs()[0].name  # 读取网络输出名称
onnx_output = sess.run([output_name], {input_name: x.cpu().numpy()})  # 读取数据进行onnx推理

# 计算转换误差
evalue = np.absolute(np.mean(torch_output_value - onnx_output))
print("\ntorch to onnx erro: ", evalue)

# 显示网络输出及结构图
session = ort.InferenceSession(
    export_onnx_file)  # 创建一个运行session，类似于tensorflow
out_r = session.run(None, {"input": np.random.rand(
    1, 3, 112, 112).astype('float32')})  # 模型运行，注意这里的输入必须是numpy类型

# 显示网络输出shape
print('网络输出shape:')
print(out_r[0].shape)

# 显示结构图
netron.start(export_onnx_file)
