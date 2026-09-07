# input: list of files
# output: list of files, containing the problem & solutions filtered by average score

import os
import json
import argparse

input_files = [
    # Update these paths to your evaluation output files
    "path/to/math500/test_pass.jsonl",
    "path/to/amc23/test_pass.jsonl",
    "path/to/olympiadbench/test_pass.jsonl",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True, help="Output path")
    return parser.parse_args()

def filter_by_avg(input_files):
    easy_set = [] # pass@1 > 0.8
    medium_set = [] # 0.4 < pass@1 < 0.8
    hard_set = []   # pass@1 < 0.4
    
    for input_file in input_files:
        if "amc23" in input_file:
            data_name = "amc23"
        elif "math500" in input_file:
            data_name = "math500"
        elif "olympiadbench" in input_file:
            data_name = "olympiadbench"
        else:
            raise ValueError(f"Unknown data name: {input_file}")
        
        file_data = []
        with open(input_file, "r") as f:
            for line in f:
                file_data.append(json.loads(line))
        
        # filter by average score
        for item in file_data:
            item["data_name"] = data_name
            avg_score = sum(item["score"]) / len(item["score"])
            if avg_score > 0.8:
                easy_set.append(item)
            elif avg_score > 0.4:
                medium_set.append(item)
            else:
                hard_set.append(item)
        
    return easy_set, medium_set, hard_set

def main():
    args = parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    easy_file = os.path.join(args.output_path, "easy.jsonl")
    medium_file = os.path.join(args.output_path, "medium.jsonl")
    hard_file = os.path.join(args.output_path, "hard.jsonl")
    
    easy_set, medium_set, hard_set = filter_by_avg(input_files)
    
    # print the number of problems in each set
    print(f"Number of problems in easy set: {len(easy_set)}")
    print(f"Number of problems in medium set: {len(medium_set)}")
    print(f"Number of problems in hard set: {len(hard_set)}")
    
    # with open(easy_file, "w") as f:
    #     for item in easy_set:
    #         f.write(json.dumps(item) + "\n")
    # with open(medium_file, "w") as f:
    #     for item in medium_set:
    #         f.write(json.dumps(item) + "\n")
    # with open(hard_file, "w") as f:
    #     for item in hard_set:
    #         f.write(json.dumps(item) + "\n")
    
def filter_by_analysis(input_files, output_file):
    ret = []
    hist = [0] * 20
    for input_file in input_files:
        with open(input_file, "r") as f:
            for line in f:
                item = json.loads(line)
                if "analysis" in item and len(item["analysis"]) > 2: # more than 2 approach
                    ret.append(item)
                    hist[len(item["analysis"])] += 1
    
    with open(output_file, "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
        
    print(f"Number of problems in {output_file}: {len(ret)}")
    print(hist)
    
    return ret
    

if __name__ == "__main__":
    input_files = [
        # Update these paths to your analysis input files
        "path/to/analysis_input_1.jsonl",
        "path/to/analysis_input_2.jsonl",
    ]
    filter_by_analysis(input_files, "path/to/output/filtered.jsonl")