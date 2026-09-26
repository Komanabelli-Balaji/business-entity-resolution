import os
import csv
import json

data_dir = "/home/balaji/Projects/entity-resolution/student_resource/dataset"
files_to_check = [
    "train/train_source1.tsv",
    "train/train_source2.tsv",
    "train/train_source3.tsv",
    "train/train_ground_truth.tsv",
    "test/test_source1.tsv",
    "test/test_source2.tsv",
    "test/test_source3.tsv"
]

report = {}

for f in files_to_check:
    path = os.path.join(data_dir, f)
    stat = os.stat(path)
    file_size_mb = stat.st_size / (1024 * 1024)
    
    with open(path, 'r', encoding='utf-8') as file:
        header_line = file.readline()
        delimiter = '\t' if '\t' in header_line else ','
        headers = [h.strip() for h in header_line.split(delimiter)]
        
        sample = []
        for i in range(5):
            line = file.readline()
            if not line: break
            sample.append([c.strip() for c in line.split(delimiter)])
            
        file.seek(0)
        file.readline() # skip header
        
        row_count = 0
        missing = {h: 0 for h in headers}
        
        match_distribution = {}
        unique_ids = set()
        duplicate_ids = 0
        
        for line in file:
            row_count += 1
            cols = [c.strip() for c in line.split(delimiter)]
            for i, c in enumerate(cols):
                if i < len(headers):
                    if not c:
                        missing[headers[i]] += 1
            
            # Record first column (usually ID)
            if cols:
                id_val = cols[0]
                if id_val in unique_ids:
                    duplicate_ids += 1
                else:
                    unique_ids.add(id_val)
                    
            if "ground_truth" in f:
                matched_ids = cols[1].split(',') if len(cols) > 1 and cols[1] else []
                num_matches = len(matched_ids)
                match_distribution[num_matches] = match_distribution.get(num_matches, 0) + 1
                
        report[f] = {
            "size_mb": round(file_size_mb, 2),
            "delimiter": r"\t" if delimiter == '\t' else delimiter,
            "headers": headers,
            "col_count": len(headers),
            "sample": sample,
            "row_count": row_count,
            "missing": missing,
            "duplicate_ids": duplicate_ids
        }
        if "ground_truth" in f:
            report[f]["match_distribution"] = match_distribution

with open("outputs/inspection_results.json", "w") as out:
    json.dump(report, out, indent=2)
print("Inspection complete.")
