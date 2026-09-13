import pandas as pd

# 构造模型参数表
data = {
    "Size": ["small", "medium", "large", "xl", "10B"],
    "d_model": [768, 1024, 1280, 2560, 4608],
    "d_ff": [3072, 4096, 5120, 10240, 12288],
    "num_layers": [12, 24, 36, 32, 50],
    "num_heads": [12, 16, 20, 32, 36]
}
df = pd.DataFrame(data)

# 打印DataFrame看效果
print(df)

# 生成LaTeX代码，不输出pandas自带索引，带竖线横线，匹配原图样式
latex_str = df.to_latex(
    index=False,
    column_format="|c|c|c|c|c|",  # 全部居中 + 竖线，和图片表格一致
    escape=False
)
print("\n==== LaTeX Code ====")
print(latex_str)