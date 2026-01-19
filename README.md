# Pic-Auto-Rotate-ENG-VLM

基于前景掩码/Alpha 的自动旋转工具。适用于 RGBA 图、或 RGB + 独立 mask 的批量纠偏；当没有 mask 时可选用 Inspyrenet 抠图作为回退。

## 工程化流程

1. **获取前景**：优先使用 RGBA 的 Alpha；其次读取外部 mask；再不行可调用 RMBG 生成临时 Alpha。
2. **掩码预处理**：阈值二值化 + 形态学闭运算，减少小孔洞/断裂。
3. **几何过滤**：取最大外轮廓，过滤面积过小/长宽比太低的样本。
4. **角度估计**：按策略计算倾角，`auto` 会按顺序尝试多个方法。
5. **阈值判断**：小角度直接跳过；超出合理范围标为 `suspicious`。
6. **图像旋转**：以估计中心旋转并扩展画布，透明填充，支持白底合成。

## 旋转策略

`--rotate-strategy` 支持：

- `top-edge`：在顶部带宽内拟合上缘直线，适合头肩/主体顶部明显的图。
- `hough`：边缘 + Hough 直线中位角，适合主体边界较直的图。
- `pca`：对前景像素做 PCA，取主轴角度，鲁棒但易受形状影响。
- `minrect`：最小外接矩形长边角度，快速但对非矩形目标敏感。
- `auto`：按 `top-edge -> hough -> pca -> minrect` 顺序依次尝试。

## 使用示例

```bash
python auto_rotate.py --input ./source --output ./output --recursive
```

有外部 mask（与原图同名，后缀 `_mask`）：

```bash
python auto_rotate.py --input ./source --output ./output --mask-dir ./masks --mask-suffix _mask
```

输出白底：

```bash
python auto_rotate.py --input ./source --output ./output --white
```

指定策略：

```bash
python auto_rotate.py --input ./source --output ./output --rotate-strategy hough
```

## RMBG 模型说明

仓库不包含 `rmbg-model/inspyrenet.safetensors`（体积较大）。如需本地抠图能力，请下载模型并放置到该路径，或使用参数 `--rmbg-model` 指定位置。

模型下载地址：https://www.modelscope.cn/models/greenCrystal/inspyrenet
