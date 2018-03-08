# -*- coding:utf-8 -*-
'''
import xlsxwriter

workbook = xlsxwriter.Workbook('temp/tttttttt.xlsx')
worksheet = workbook.add_worksheet()

with open('temp/results.txt', 'r') as infile:
    for line in infile.readlines():
        line = line.strip('\n')
        img_path, text_seq = line.split(' ')
        img_name, _ = img_path.split('/')[-1].split('.')
        row, col = img_name.split('_')
        worksheet.write(int(row), int(col), text_seq.decode('utf-8'))


workbook.close()

'''
