## 总述

## PLM

### 模型

当前为全部手动下载到本地的指定路径

```
hf download google-bert/bert-base-chinese
hf download ethanyt/guwenbert-base
hf download SIKU-BERT/sikubert
```



### 代码

#### 运行顺序

1. 3-bert-original.py：原始训练测试
2. 3-bert-adapter.py：hypernetwork的训练
3. 3-bert-ad-train.py：+hypernetwork的训练测试
4. 3-bert-cl-train.py：+对比学习的训练测试

#### 注意事项

1. 有一些参数，如batch_size, epochs的值需要调整
2. 需要修改所有路径



## LLM

1. requirements1 版本的环境 llama-factory 会和 xinference 有冲突，每次运行的时候需要换不同的包版本
2. requirements2 版本环境无冲突，但不一定能适配硬件
3. 所有路径最好都用绝对路径

### 代码

#### 运行顺序

1. 4-LLM-main.py：原始测试
2. 4-LLM-LoRA.ipynb：lora方法生成适配器 (合并后直接用4-LLM-main.py做测试即可)
3. 4-LLM-adapter.py: 各个模型的`朝代感知适配器`生成 (需要确保模型包含tokenizer文件)
4. 4-LLM-main-integrated.py：各个模型加入`朝代感知适配器`的测试
5. 4-LLM-unified-wrapper：无需手动运行

```
对于一个模型：
原始模型：
1. 先启动xinference，注册好模型后就可以运行4-LLM-main.py
2. 运行完成后会有一个结果文件,运行4-LLM-report.py得到分类报告

LoRA:
1. 根据4-LLM-LoRA.ipynb得到配置文件，修改命令中的配置文件路径，运行完两个命令之后就能得到合并的模型
2. 再使用xinference重新注册合并后的新模型
3. 运行4-LLM-main.py得到结果文件
4. 运行4-LLM-report.py得到分类报告

Hypernetwork:
1. 运行4-LLM-main.py,得到.pt文件
2. 运行4-LLM-main-integrated.py得到结果文件
3. 运行4-LLM-report.py得到分类报告
```



### 安装

```
hf download mergekit-community/Qwen3-1.5B-Instruct --local-dir /root/autodl-tmp/models/Qwen3-1.5B-Instruct (这个可能得手动在网站下载，用命令行可能会报错)
hf download Qwen/Qwen2.5-7B-Instruct


hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
hf download deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local-dir /root/autodl-tmp/models/DeepSeek-R1-Distill-Llama-8B


以下llama的两个模型需要现在hf上申请权限
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir /root/autodl-tmp/models/Llama-3.2-3B-Instruct
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir /root/autodl-tmp/models/Llama-3.1-8B-Instruct
如果申请不到就用modelscope, 见https://github.com/SesameAILabs/csm/issues/99
modelscope download --model LLM-Research/Llama-3.2-3B-Instruct --local_dir /root/autodl-tmp/models/Llama-3.2-3B-Instruct
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --local-dir /root/autodl-tmp/models/Llama-3.1-8B-Instruct


gemma模型需要先填写个表
hf download google/gemma-7b --local-dir /root/autodl-tmp/models/gemma-7b


hf download openai-community/gpt2 --local-dir /root/autodl-tmp/models/gpt2(1.24b)
```

#### 命令行登录

```python
1. 登录 Hugging Face
    hf auth login
2. 然后输入你的 token
    获取 Token：https://huggingface.co/settings/tokens
    选择：
    ✔ Read access
```

如果下不了就上网页下载





### 模型设置

1. 启动

   ```
   xinference-local --host 0.0.0.0 --port 9997
   ```

2. 注册模型：如果需要用vLLM运行模型，则在注册时不同量化模型需要填写 Quantization 的内容不同，[参考](https://inference.readthedocs.io/zh-cn/stable/getting_started/installation.html)

3. 启动时如果出现以下错误

   ```
   File "/root/miniconda3/lib/python3.12/site-packages/xinference/model/llm/vllm/core.py", line 614, in load engine_args = AsyncEngineArgs( ^^^^^^^^^^^^^^^^^
   TypeError: [address=0.0.0.0:45641, pid=2189] AsyncEngineArgs.__init__() got an unexpected keyword argument 'swap_space'
   ```

   可修改：`site-packages/xinference/model/llm/vllm/core.py`

   ```
   1. 找到包含'swap_space'
   2. 注释掉即可
   ```



### LoRA

#### 安装LLama-factory

```
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```



#### 配置过程

1. 见 4-LLM-LoRA.ipynb 文件
2. 内容基本改路径和名称就行



#### 运行

1. 最好在llama factory文件夹内运行命令，不然会找不到文件
2. 过程中会产生很多中间文件占内存，如 checkpoint 内容可及时删除
3. 合并后利用xinference注册-运行即可
