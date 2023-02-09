import os
import json

m = 1
n = 2
head = -200
tail = 250

input_json_folder_name = 'vocal_output/'
output_json_folder_name = 'vocal_after_21/'

input_json_folder_path = os.path.join(os.path.dirname(__file__), input_json_folder_name)
output_json_folder_path = os.path.join(os.path.dirname(__file__), output_json_folder_name)

stdout_path = os.path.join(os.path.dirname(__file__), 'stdout.txt')
with open(stdout_path, 'w', encoding='utf-8') as stdout_file:
  for inout_json_file_name in os.listdir(input_json_folder_path)[0:]:
    print(inout_json_file_name, file=stdout_file)

    input_json_file_path = os.path.join(input_json_folder_path, inout_json_file_name)
    with open(input_json_file_path, 'r', encoding='utf-8') as input_json_file:
      lines = json.load(input_json_file)
    # with open(input_json_file_path, 'r', encoding='utf-8') as input_json_file:
    #   old_lines = json.load(input_json_file)

    for i in range(len(lines)):
      for j in range(len(lines[i]['l'])):
        if i == j == 0:
          lines[i]['l'][j]['s'] = max(0, lines[i]['l'][j]['s'] + head)
        if j == len(lines[i]['l']) - 1:
          if i == len(lines) - 1:
            if lines[i]['l'][j]['e'] == 0:
              lines[i]['l'][j]['e'] = lines[i]['l'][j]['s'] + tail
            else:
              lines[i]['l'][j]['e'] += tail
          else:
            if lines[i+1]['l'][0]['s'] == 0:
              lines[i+1]['l'][0]['s'] = lines[i]['l'][j]['e']
            elif lines[i]['l'][j]['e'] == 0:
              lines[i]['l'][j]['e'] = lines[i+1]['l'][0]['s']
            else:
              lines[i]['l'][j]['e'] = lines[i+1]['l'][0]['s'] = round((lines[i]['l'][j]['e'] * m + lines[i+1]['l'][0]['s'] * n) / (m + n))
        else:
          if lines[i]['l'][j+1]['s'] == 0:
            lines[i]['l'][j+1]['s'] = lines[i]['l'][j]['e']
          elif lines[i]['l'][j]['e'] == 0:
            lines[i]['l'][j]['e'] = lines[i]['l'][j+1]['s']
          else:
            lines[i]['l'][j]['e'] = lines[i]['l'][j+1]['s'] = round((lines[i]['l'][j]['e'] * m + lines[i]['l'][j+1]['s'] * n) / (m + n))

      lines[i]['s'] = lines[i]['l'][0]['s']
      lines[i]['e'] = lines[i]['l'][-1]['e']
    # for i in range(len(lines)):
    #   for j in range(len(lines[i]['l'])):
    #     print(lines[i]['l'][j]['d'], old_lines[i]['l'][j]['s'], old_lines[i]['l'][j]['e'], lines[i]['l'][j]['s'], lines[i]['l'][j]['e'], file=stdout_file)

    output_json_file_path = os.path.join(output_json_folder_path, inout_json_file_name)
    with open(output_json_file_path, 'w', encoding='utf-8') as output_json_file:
      json.dump(lines, output_json_file, ensure_ascii=False)
