import json
from openai import OpenAI

client = OpenAI(api_key="REMOVED_CREDENTIAL")

def use_llm(text):
    prompt = f"""你是一个水保方案审核专家，同时也是一个逻辑大师。现有另一组不太靠谱的校核团队针对一个水保方案找出了一系列问题和不足，其中有很多根本不成立。
    
请你检查这些问题，仔细思考，筛选掉满足以下任一条件的问题，只保留最有价值的30个问题：
1. 显然不成立、不合逻辑的问题
2. 过于细节，实际上无伤大雅的问题
3. 与表格格式相关
4. 与公式相关

直接输出筛选过后剩余的30个问题的编号。

【待筛选的问题集】：
{text}

只输出编号序列，不要输出任何其他内容和解释。
"""

    response = client.chat.completions.create(
        model="o3-2025-04-16", # 换会思考的模型gpt-4.1//gpt-5-2025-08-07
        messages=[{"role": "user", "content": prompt}],
        # temperature=0
    )

    result = response.choices[0].message.content
    return result

result = use_llm(output_text)