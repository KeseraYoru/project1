"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        # 拆出三维：batch、token 数（50，含 CLS）、每个 token 的特征长度 768
        b, n, d = x.shape
        # 第 1 步 reshape：把特征轴 768 拆成（12 个头，每头 64 维）→ [B,N,12,64]
        #   头 0 拿的是连续的前 64 列，头 11 拿最后 64 列，不是「每隔 12 个取一个」
        # 第 2 步 transpose(0,2,1,3)：把头轴挪到 token 轴之前 → [B,H,N,64]
        #   目的：后面 q @ kᵀ 只对最后两维做矩阵乘，(B,H) 会被当成「批」一次算完
        # 两步都不复制内存，只改 shape 和步长，属于零拷贝的 view
        return x.reshape(b, n, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        # 缩放因子 1/√64 = 0.125：点积是 64 项相加，量级约 √64；
        #   不缩小会让 softmax 过饱和（一个接近 1、其余全 0），梯度就没了
        # 必须包成 np.float32：float64 标量会把整个张量升级成 float64，dtype 检查不过
        scale = np.float32(1.0 / np.sqrt(self.head_dim))
        # k 只交换最后两维：[B,H,N,64] → [B,H,64,N]，
        #   这样 [N,64] @ [64,N] 才能乘出 N×N 的「谁该关注谁」分数表
        # 注意不能写 k.T：四维数组的 .T 会把 4 个轴全反转，连 batch 和头一起翻掉
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        # softmax 保证每行加起来等于 1，所以这一步就是「按注意力百分比
        #   对 50 个 v 做加权平均」，结果的幅度不会爆炸
        context = coefficients @ v                              # [B,H,N,64]
        # transpose 把头轴还原到 token 之后 → [B,N,12,64]，与 TODO-02 严格互逆
        # reshape 再合并最后两维 → [B,N,768]；头 0 的 64 维必须在最前面，
        #   才和权重里「特征列按头依次排列」的约定对得上
        merged = context.transpose(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[1], self.width)                  # [B,N,768]

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        # 768→768 的可学习混合，决定 12 个头的结论怎么综合；
        #   权重按 [out,in] 存储，所以右乘时要转置
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        # 残差：保留「未归一化」的原值 x，只把注意力算出的增量叠上去；
        #   这样堆 12 层也不会把信息冲掉（最坏情况增量≈0，原值仍在）
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        # 先升维 768→3072 当「草稿纸」，过精确 GELU（助手已提供）；权重 [out,in] 要转置
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)   # [B,N,3072]
        # 再降回 768，得到这一层的增量；形状必须与原值一致才能相加
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias      # [B,N,768]
        # 第二条残差：residual + MLP 增量（本模型没有 LayerScale 参数）
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        # 拆出四维：batch、通道、高、宽（输入已被预处理成 224×224）
        b, c, h, w = images.shape
        # 每块边长 32，由 conv1 的卷积核决定；3×32×32 = 3072 就是每行的长度
        p = 32
        # 整除：竖着 7 块、横着 7 块（224 // 32 = 7；注意 // 才是整数除法）
        nh, nw = h // p, w // p
        # 第 1 步 reshape：把 224 各拆成「块号 7」+「块内位置 32」
        #   → [B,C,7,32,7,32]，此时两个块轴被 p 隔开了
        grid = images.reshape(b, c, nh, p, nw, p)
        # 第 2 步 transpose：只调轴顺序、不搬数据，把两个块轴挪到一起
        #   [B,C,7,32,7,32] → [B,7,7,C,32,32]
        # 第 3 步 reshape：前两轴合成 49（先左→右、再上→下），
        #   后三轴按 C→H→W 摊平成 3072
        return grid.transpose(0, 2, 4, 1, 3, 5).reshape(b, nh * nw, c * p * p)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        # 3072→768 的投影，等价于 stride=32 的卷积（块之间不重叠）；
        #   原 conv1 没有 bias，所以这里也不加
        projected = patches @ self.patch_weight.T   # [B,49,3072]@[3072,768]

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        # 沿 token 轴拼接：CLS 放第 0 行，49 个 patch 跟在后面 → [B,50,768]
        #   CLS 是学出来的空位（记录员），12 层之后它身上就是整张图的摘要
        tokens = np.concatenate([cls, projected], axis=1)   # [B,50,768]
        # 加位置编码：注意力本身分不清「左上」和「右下」（打乱顺序只是换个排列），
        #   必须显式告诉它谁坐哪；position 是 [50,768]，广播会自动复制到 B 张图上，
        #   其中第 0 行对应 CLS，第 1~49 行对应块 0~48
        return tokens + self.position                        # [50,768] 广播到 B

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        # TODO-06: CLS selection, post-Norm and projection.
        # tokens[:, 0, :] picks the CLS row of every image -> [B,768].
        # 只取第 0 行（CLS）：它专门被训练来汇总整张图，比平均 49 个 patch 更合适
        cls_row = self.post_norm(tokens[:, 0, :])     # [B,768]
        # 768→512：visual.proj 本身就是 [in,out]，是唯一不需要转置的权重
        output = cls_row @ self.projection            # [768,512] 已是 in,out
        # 缩成单位长度：CLIP 用余弦相似度比较图文，单位化后点积就直接等于余弦
        return unit_vector(output)
