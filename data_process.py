from utils import *
import json

# generate tokens
with open('./dataset/api_info_manual.json') as json_file:
    api_infos=json.load(json_file)
extra_tokens = []
for key in api_infos:
    func_name = api_infos[key]['function_name']
    func_name = func_name.split('(')[0]
    func_name=func_name.replace('scanpy', 'sc')
    extra_tokens.append(func_name)
extra_tokens = extra_tokens[:72]
print(extra_tokens)
save_obj(extra_tokens,'./dataset/processed_data/function_tokens.list')

# process trainset
task_set = load_obj('./dataset/task_generation002_reph_fix.list')
document_pool = []
task_pool = []
for i in range(71):
    document = api_infos[list(api_infos.keys())[i]]
    usage_tips = parse_response(task_set[i][0])[0]['Output']['tips']
    if 'pp.' in usage_tips and 'sc.pp.' not in usage_tips:
        usage_tips = usage_tips.replace('pp.','sc.pp.')
    if 'tl.' in usage_tips and 'sc.tl.' not in usage_tips:
        usage_tips = usage_tips.replace('tl.', 'sc.tl.')
    if ' neighbors()' in usage_tips and ' sc.pp.neighbors()' not in usage_tips:
        usage_tips=usage_tips.replace(' neighbors()',' sc.pp.neighbors()')
    if ' diffmap()' in usage_tips and ' sc.tl.diffmap()' not in usage_tips:
        usage_tips = usage_tips.replace(' diffmap()', ' sc.pp.diffmap()')
    if ' dpt()' in usage_tips and ' sc.tl.dpt()' not in usage_tips:
        usage_tips = usage_tips.replace(' dpt()', ' sc.pp.dpt()')
    description = document['description']
    parameters = document['Parameters']
    if "Returns" in document:
        returns = document['Returns']
        document_pool.append(f'A brief description about returns or result of function {extra_tokens[i]}:{returns}')
    document_pool.append(f'A brief description of function {extra_tokens[i]}:{description}')
    document_pool.append(f'A brief description about parameters of function {extra_tokens[i]}:{parameters}')
    document_pool.append(f'Some tips about using function {extra_tokens[i]}:{usage_tips}')
    for task in task_set[i]:
        task_dict = parse_response(task)[0]
        instruction = task_dict['Instruction']
        code = task_dict['Output']['code']
        task_pool.append(
            f"""
            Write a python script using scanpy to fulfill the following task.
            Task: {instruction}
            Script: 
            {code}
            """
        )
save_obj(document_pool,'./dataset/processed_data/train_document')
save_obj(task_pool,'./dataset/processed_data/train_task')




