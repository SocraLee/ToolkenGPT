import pickle
def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def parse_response(response):
    response = response.strip()

    result = []
    parts = response.split("Instruction:")[1:]
    # 遍历列表中的每个元素，每个元素都是一个instruction和output的组合
    for part in parts:
        parse_result = {}
        subparts = part.split("Output:")
        instruction = subparts[0].strip()
        parse_result['Instruction']=instruction

        s = subparts[1]
        s = s.strip()[1:-1].strip()
        parts = s.split("\n")
        # 创建一个空字典来存储键值对
        output = {}
        key = ""
        value = ""
        # 遍历每个部分
        for part in parts:
            # 如果该部分包含冒号，那么它是一个键
            if ":" in part:
                # 如果之前已经有一个键值对，那么将其添加到字典中
                if key and value:
                    output[key] = value.strip()
                # 将新的键存储起来，准备获取其对应的值
                key = part.replace(":", "").strip()
                value = ""
            else:
                # 如果该部分不包含冒号，那么它是一个值的一部分
                value += part + "\n"
        # 添加最后一个键值对到字典中
        if key and value:
            output[key] = value.strip()

        parse_result['Output']=(output)
        result.append(parse_result)
    # 返回result
    return result