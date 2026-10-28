"""Pure-function regression tests for V11.2 review and time labels."""
import ast
import pathlib
import unittest
import numpy as np
SCRIPT=pathlib.Path(__file__).resolve().parents[1]/'222.py'
text=SCRIPT.read_text(encoding='utf-8')
parsed=ast.parse(text)
wanted={'v112_is_informational_notice','v112_split_review_items',
        'v9_sample_row','v9_camera_seconds','v9_frame_index'}
functions=[node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name in wanted]
assert {x.name for x in functions} == wanted
ns={'np':np}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(SCRIPT),'exec'),ns)

class TestV112(unittest.TestCase):
    def test_review_vs_notice_do_not_disappear(self):
        items=[{'紀錄類型':'骨架品質提醒（仍可採信）','樣本編號':3},
               {'紀錄類型':'骨架品質待複核（有候選 RULA 分數）','樣本編號':13},
               {'評分狀態':'排除，不自動計入 RULA','樣本編號':15},
               {'紀錄類型':'其他未知紀錄','樣本編號':20}]
        review,notice=ns['v112_split_review_items'](items)
        self.assertEqual([x['樣本編號'] for x in review],[13,15,20])
        self.assertEqual([x['樣本編號'] for x in notice],[3])
        self.assertEqual(len(review)+len(notice),len(items))
    def test_time_maps_different_camera_offsets_and_positive_elapsed(self):
        m={'Front':{'fps':30.,'count':900},'Right':{'fps':25.,'count':900},'Left':{'fps':29.,'count':900}}
        a={'Front':19.5,'Right':19.75,'Left':19.75}
        row=ns['v9_sample_row'](-17.5,m,a,list(m),3,analysis_start=-19.5)
        self.assertEqual(row['時間(秒)'],-17.5)
        self.assertEqual(row['分析經過時間(秒)'],2.)
        self.assertEqual(row['Front影片時間(秒)'],2.)
        self.assertEqual(row['Right影片時間(秒)'],2.25)
        self.assertEqual(row['Right影格索引'],56)
    def test_non_actionable_photo_export_is_preserved(self):
        # Source retains all photos and exports informational records separately.
        self.assertIn('v8_zip_rejected_photos(zf,rejected_photos)',text)
        self.assertIn('Diagnostics/Quality_Notices.csv',text)
        self.assertIn('Diagnostics/Rejected_Samples_Detail.csv',text)
    def test_main_scoring_unchanged(self):
        original=pathlib.Path('/mnt/data/222_rula_v11_1.py').read_text(encoding='utf-8') if pathlib.Path('/mnt/data/222_rula_v11_1.py').exists() else None
        if original:
            for fn in ('compute_rula_full','get_wrist_score','get_neck_score','v11_quality_from_snapshot'):
                def grab(src):
                    tree=ast.parse(src)
                    return ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==fn),include_attributes=False)
                self.assertEqual(grab(text),grab(original),fn)

if __name__=='__main__':unittest.main()
