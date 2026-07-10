import json
import ast
import pycodestyle
from tqdm import tqdm
import os
import argparse
import signal
import warnings
import multiprocessing as mp
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

from autograder import Autograder


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--file", default=None, nargs="+", help="One or more _formatted.jsonl filenames to process (optional)")
parser.add_argument("--test_files_dir", default=None, help="Directory with doctest files named {semester}_{question_name}.py")
parser.add_argument("--test_class_dir", default=None, help="Optional directory with question-class JSON files")
parser.add_argument("--question", action="append", default=None, help="Optional question filter; can be repeated")
args = parser.parse_args()

INPUT_DIR  = args.input_dir
OUTPUT_DIR = args.output_dir
TEST_FILES_DIR = args.test_files_dir
TEST_CLASS_DIR = args.test_class_dir


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore", message=".*optimum is not installed.*")
style_guide     = pycodestyle.StyleGuide(quiet=True)

def _timeout_handler(signum, frame):
    raise TimeoutError("entry timed out")

test_class_map = {}
if TEST_CLASS_DIR and os.path.isdir(TEST_CLASS_DIR):
    for fname in os.listdir(TEST_CLASS_DIR):
        if not fname.endswith(".json"):
            continue
        cls = os.path.splitext(fname)[0]
        path = os.path.join(TEST_CLASS_DIR, fname)
        with open(path) as tf:
            questions = json.load(tf)
        for qn in questions:
            test_class_map[qn] = cls



feature_cache = {}

class CaptureReport(pycodestyle.BaseReport):
    def __init__(self, options):
        super().__init__(options)
        self.errors = []

    def error(self, line_number, offset, text, check):
        code = super().error(line_number, offset, text, check)
        if code:
            self.errors.append({
                "line": line_number,
                "col": offset,
                "msg": text
            })
        return code

def count_pep8_violations(code):
    if not code.endswith('\n'):
        code += '\n'

    wrapper = "def _tmp_func():\n"
    for line in code.strip().splitlines():
        wrapper += f"    {line.rstrip()}\n"

    report = CaptureReport(style_guide.options)
    checker = pycodestyle.Checker(
        lines=wrapper.splitlines(),
        report=report
    )
    checker.check_all()
    return {
        "count": checker.report.total_errors,
        "messages": report.errors
    }


def get_ast_tree_metrics(code):
    try:
        tree = ast.parse(code.strip())
    except:
        return -1, -1, -1, -1

    max_depth = node_count = total_children = non_leaf = 0
    widths = {}
    def visit(node, depth=0):
        nonlocal max_depth, node_count, total_children, non_leaf
        max_depth = max(max_depth, depth)
        widths[depth] = widths.get(depth, 0) + 1
        node_count += 1
        children = list(ast.iter_child_nodes(node))
        total_children += len(children)
        if children:
            non_leaf += 1
        for c in children:
            visit(c, depth+1)
    visit(tree)
    avg_branch = total_children / non_leaf if non_leaf else 0
    max_width  = max(widths.values()) if widths else 0
    return max_depth, max_width, node_count, avg_branch

def extract_features(code: str):
    key = code.strip()
    if key.upper() == "NONE" or key == "":
        return None
    if key in feature_cache:
        return feature_cache[key]

    cleaned = code.strip()
    depth, width, nodes, branch = get_ast_tree_metrics(code)

    feat = {
        "loc": cleaned.count("\n") + 1,
        "char_count": len(cleaned),
        "pep8_violations": count_pep8_violations(cleaned),
        "ast_depth": depth,
        "ast_width": width,
        "ast_node_count": nodes,
        "ast_avg_branching": branch
    }
    feature_cache[key] = feat
    return feat

if __name__ == "__main__":
    file_list = args.file if args.file else os.listdir(INPUT_DIR)
    for filename in file_list:
        # if not filename.endswith("_formatted.jsonl"):
        if not filename.endswith("_formatted.jsonl"):
            continue


        in_path  = os.path.join(INPUT_DIR,  filename)
        out_path = os.path.join(
            OUTPUT_DIR,
            filename.replace("_formatted", "_with_features")
        )

        updating_existing = os.path.exists(out_path)
        if updating_existing:
            print(f"Updating existing {out_path}; target-question rows will be regenerated.")



        with open(in_path) as f:
            data = [json.loads(line) for line in f if line.strip()]

        target_questions = set(args.question or [])
        if target_questions:
            data = [row for row in data if row["question_name"] in target_questions]

        # # Sample 15% of (student, question) streams per question
        # import random
        # random.seed(42)
        # from collections import defaultdict
        # streams = defaultdict(list)
        # for row in data:
        #     streams[(row["student_id"], row["question_name"])].append(row)
        # streams_by_q = defaultdict(list)
        # for (sid, qname), rows in streams.items():
        #     streams_by_q[qname].append((sid, qname))
        # data = []
        # for qname, pairs in streams_by_q.items():
        #     k = max(1, int(len(pairs) * 0.15))
        #     sampled_pairs = set(random.sample(pairs, k))
        #     for (sid, qn), rows in streams.items():
        #         if (sid, qn) in sampled_pairs:
        #             data.extend(rows)

        # data = data[:10]



        results = []

        for row in tqdm(data, desc=filename):
            tc = test_class_map.get(row.get("question_name"), None)
            if row["question_name"] == "Mint" or row.get("is_processed") is False:
                results.append(row)
                continue

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)

            try:
                tc = test_class_map.get(row["question_name"], None)
                test_path = None
                if TEST_FILES_DIR:
                    test_file = f"{row['semester']}_{row['question_name']}.py"
                    test_path = os.path.join(TEST_FILES_DIR, test_file)

                for side in ["synthetic", "gt"]:
                    key = f"{side}_code_block"
                    raw = row.get(key, None)

                    # Debug print for missing fields
                    if raw is None:
                        print(
                            "Missing code_block:",
                            f"file={filename}",
                            f"question={row.get('question_name')}",
                            f"side={side}",
                            f"row_index={row.get('row_index')}",
                            f"value={raw}",
                        )

                    code = (raw or "").strip()

                    if code == "" or code.upper() == "NONE":
                        row[f"{key}_features"] = None
                        row[f"{key}_autograder"] = None
                    else:
                        row[f"{key}_features"] = extract_features(code)
                        if test_path and os.path.exists(test_path):
                            row[f"{key}_autograder"] = Autograder.grade_submission(code, test_path)
                        else:
                            row[f"{key}_autograder"] = None

                new_row = {"test_class": tc}
                for k, v in row.items():
                    new_row[k] = v
                results.append(new_row)

            except TimeoutError:
                for side in ["gt", "synthetic"]:
                    key = f"{side}_code_block"
                    row[f"{key}_features"] = None
                    row[f"{key}_autograder"] = None

                new_row = {"test_class": tc}
                for k, v in row.items():
                    new_row[k] = v
                results.append(new_row)

            finally:
                signal.alarm(0)

        # --- Merge with existing file ---
        existing_rows = []
        if os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    row = json.loads(line)
                    if target_questions and row.get("question_name") not in target_questions:
                        existing_rows.append(row)

        all_rows = existing_rows + results

        with open(out_path, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")

        action = "Updated" if updating_existing else "Created"
        scope = target_questions if target_questions else "all questions"
        print(f"{action} {out_path}: replaced {len(results)} rows for {scope}, "
              f"kept {len(existing_rows)} others.")
