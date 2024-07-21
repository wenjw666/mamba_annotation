from dataclasses import dataclass, field


@dataclass
class MambaConfig:

    d_model: int = 2560 # 模型的维度
    d_intermediate: int = 0 
    n_layer: int = 64 # Mamba 的层数, 即堆叠的 Mamba Block 的数量
    vocab_size: int = 50277 # 词表大小, 即模型的输入/输出词表中包含的唯一词(或 subword)的数量
    ssm_cfg: dict = field(default_factory=dict) # State Space Model(SSM)的配置参数, 以字典的形式给出。这包括状态矩阵 A, B, C 的维度, 时间步长 dt 的范围等。默认为空字典
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True #是否使用 RMSNorm 替代传统的 LayerNorm
    residual_in_fp32: bool = True # 是否在训练时以 float32 格式存储残差连接。这可以减少数值误差, 提高模型稳定性。这里设为 True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
