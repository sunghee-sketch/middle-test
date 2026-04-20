import json

path = '/Users/choisunghee/Desktop/middle-test/v9_optuna/melting_point_v9.ipynb'
try:
    with open(path, 'r', encoding='utf-8') as f: 
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code' and '# OOF stack' in ''.join(cell.get('source', [])):
            clean_source = []
            for line in cell['source']:
                if 'print(f"\n' == line: continue
                if line.startswith('★ Stacking'): continue
                if '")\n' == line: continue
                clean_source.append(line)
            
            final_source = []
            for line in clean_source:
                final_source.append(line)
                if 'stack_cv_scores = cross_val_score' in line:
                    code = 'print(f"\\n★ Stacking 메타 모델 CV 평균: R²={np.mean(stack_cv_scores):.4f} ± {np.std(stack_cv_scores):.4f}\\n")\n'
                    final_source.append(code)
            
            cell['source'] = final_source

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Print block fixed safely.")
except Exception as e:
    print("Error:", e)
