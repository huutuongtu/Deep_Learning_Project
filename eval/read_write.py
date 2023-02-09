import os
import csv
import json

input_csv_file = 'song_output_base.csv'
drive_folder_path = '/home/tuht/DL/PL/song_output'
input_json_folder_name = 'new_labels_json'
output_json_folder_name = 'vocal_output'

csv_content_column = 2
csv_path_column = 1

stdout_path = os.path.join(os.path.dirname(__file__), 'stdout.txt')
input_csv_path = os.path.join(os.path.dirname(__file__), input_csv_file)

punctuations = [',', '.', '?', '!', '"', "'", '“', '”', '…']

def strip_punctuation(s):
  for i in range(len(s)):
    if s[i] in punctuations:
      i += 1
    else:
      break

  for j in range(-1, -len(s)-1, -1):
    if s[j] in punctuations:
      j -= 1
    else:
      break

  return s[i:j+1+len(s)]

def subtract_list(l1, l2):
  new_l1_indices = []
  j = 0
  for i in range(len(l1)):
    if j < len(l2) and l1[i] == l2[j]:
      j += 1
      continue
    else:
      new_l1_indices.append(i)

  return [l1[i] for i in new_l1_indices]

def sub_index(l1, l2):
  sub_indices = [None] * len(l2)

  j = 0
  for i in range(len(l2)):
    while 1:
      if j >= len(l1):
        return sub_indices
      if l2[i] == l1[j]:
        sub_indices[i] = j
        j += 1
        break
      else:
        j += 1

  return sub_indices

def lcs(l1, l2, m, n):
  L = [[0 for x in range(n+1)] for x in range(m+1)]

  for i in range(m+1):
    for j in range(n+1):
      if i == 0 or j == 0:
        L[i][j] = 0
      elif l1[i-1] == l2[j-1]:
        L[i][j] = L[i-1][j-1] + 1
      else:
        L[i][j] = max(L[i-1][j], L[i][j-1])

  index = L[m][n]
  lcs = [""] * index
  i = m
  j = n

  while i > 0 and j > 0:
    if l1[i-1] == l2[j-1]:
      lcs[index-1] = l1[i-1]
      i -= 1
      j -= 1
      index -= 1
    elif L[i-1][j] > L[i][j-1]:
      i -= 1
    else:
      j -= 1

  return lcs

rows = []
with open(stdout_path, 'w', encoding='utf-8') as stdout_file:
  with open(input_csv_path, 'r', encoding='utf-8') as input_csv_file:
    csv_reader = csv.reader(input_csv_file, delimiter=',')

    for row in csv_reader:
      rows.append(row)

  for row in rows[1:]:
    inout_json_file_name = row[csv_path_column]
    assert(inout_json_file_name.startswith(drive_folder_path))
    assert(inout_json_file_name.endswith('.npy'))
    inout_json_file_name = inout_json_file_name[len(drive_folder_path):-3] + 'json'

    print(inout_json_file_name, file=stdout_file)

    content = row[csv_content_column]
    assert(content.startswith('['))
    assert(content.endswith(']'))
    content = content[1:-1]
    words_se = content.split('), ')
    assert(words_se[-1].endswith(')'))
    words_se[-1] = words_se[-1][:-1]
    for i in range(len(words_se)):
      word = words_se[i]
      new_word = []
      j = word.find('\t')
      new_word.append(word[0:j])
      j = word.find('[')
      new_word.extend([x.strip() for x in word[j+1:].split(',')])
      words_se[i] = new_word

    input_json_path = os.path.join(os.path.dirname(__file__), f'{input_json_folder_name}\\{inout_json_file_name}')
    with open(input_json_path, 'r', encoding='utf-8') as input_json_file:
      lines = json.load(input_json_file)

    json_words = []
    for line in lines:
      for word in line['l']:
        json_words.append(word['d'])

    csv_words = [word[0] for word in words_se]
    lcs_words = lcs([strip_punctuation(w).lower() for w in json_words], csv_words, len(json_words), len(csv_words))
    not_common_json_words =  subtract_list([strip_punctuation(w).lower() for w in json_words], lcs_words)
    common_json_word_sub_indices = sub_index([strip_punctuation(w).lower() for w in json_words], lcs_words)
    if len(not_common_json_words) > 0:
      print('  json:', not_common_json_words, file=stdout_file)
    not_common_csv_words = subtract_list(csv_words, lcs_words)
    common_csv_word_sub_indices = sub_index(csv_words, lcs_words)
    if len(not_common_csv_words) > 0:
      print('  csv:', not_common_csv_words, file=stdout_file)

    i = 0
    for line in lines:
      for word in line['l']:
        if i >= len(lcs_words):
          break
        if strip_punctuation(word['d']).lower() == lcs_words[i]:
          word['s'] = int(words_se[common_csv_word_sub_indices[i]][1])
          word['e'] = int(words_se[common_csv_word_sub_indices[i]][2])
          i += 1

      for word in line['l']:
        if word['s'] != 0:
          line['s'] = word['s']
          break

      for word in line['l'][::-1]:
        if word['e'] != 0:
          line['e'] = word['e']
          break

    output_json_path = os.path.join(os.path.dirname(__file__), f'{output_json_folder_name}\\{inout_json_file_name}')
    with open(output_json_path, 'w', encoding='utf-8') as output_json_file:
      json.dump(lines, output_json_file, ensure_ascii=False)
