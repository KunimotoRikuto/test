# encoding:utf-8

import codecs

path = "\\\\S0fff001\\9共有フォルダ\\P0統計解析\\部門内\\10_テーマ別\\E1_営業活動\\170627_匿名化サンプル\\kunimoto"

# Shift_JIS ファイルのパス
shiftjis_csv_path = path + '\\ldates\\npwAB_trans.csv'
# UTF-8 ファイルのパス
utf8_csv_path = path + '\\ldates\\npwAB_trans_en.csv'

# 文字コードを utf-8 に変換して保存
fin = codecs.open(shiftjis_csv_path, "r", "shift_jis")
fout_utf = codecs.open(utf8_csv_path, "w", "utf-8")
for row in fin:
    fout_utf.write(row)
fin.close()
fout_utf.close()