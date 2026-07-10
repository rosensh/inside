import os, json, time, argparse
from tqdm import tqdm
import concurrent.futures
from dotenv import load_dotenv

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from litellm import completion
from openai import OpenAI
from transformers import StoppingCriteria, StoppingCriteriaList

from utils import *

load_dotenv()

# ---------------------
# Load config
# ---------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--index", type=int, default=None, help="Optional index to print a single input/output")
args = parser.parse_args()

with open(args.config, "r") as f:
    config_data = json.load(f)

class ScriptArguments:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

script_args = ScriptArguments(config_data)

# ---------------------
# Prepare dataset
# ---------------------
def prepare_dataset(script_args: ScriptArguments):
    dataset = load_dataset("json", data_files=[script_args.input_data_path], split="train")

    # Optional filtering using filter_questions (dict of hw -> [question_names])
    if hasattr(script_args, "filter_questions") and script_args.filter_questions:
        valid_hw_qnames = script_args.filter_questions  # dict
        dataset = dataset.filter(
            lambda x: x["INPUT"]["assignment_name"] in valid_hw_qnames
                    and x["INPUT"]["question_name"] in valid_hw_qnames[x["INPUT"]["assignment_name"]]
        )


    # Flatten INPUT fields into top-level keys
    def flatten_input(x):
        return {**x, **x["INPUT"]}

    dataset = dataset.map(flatten_input, load_from_cache_file=False)

    if script_args.experiment_name == "exp1":
        dataset = dataset.map(lambda x: process_input_exp1(x, script_args.prompt_template), load_from_cache_file=False)
        dataset = dataset.map(process_output_exp1, load_from_cache_file=False)
    elif script_args.experiment_name == "exp2":
        dataset = dataset.map(lambda x: process_input_exp2(x, script_args.prompt_template), load_from_cache_file=False)
        dataset = dataset.map(process_output_exp2, load_from_cache_file=False)
    elif script_args.experiment_name == "exp2_1":
        dataset = dataset.map(process_input_exp2_1, load_from_cache_file=False)
        dataset = dataset.map(process_output_exp1, load_from_cache_file=False)
    elif script_args.experiment_name == "exp2_2":
        dataset = dataset.map(process_input_exp2_2, load_from_cache_file=False)
        dataset = dataset.map(process_output_exp1, load_from_cache_file=False)
    else:
        raise ValueError(f"Unsupported experiment: {script_args.experiment_name}")

    dataset = dataset.remove_columns(["INPUT", "OUTPUT"])
    return dataset



# ---------------------
# Load dataset
# ---------------------
dataset = prepare_dataset(script_args)
print(f"Loaded dataset with {len(dataset)} examples")

# ---------------------
# Filter out already-generated rows
# ---------------------
existing_keys = set()

def generation_key(row):
    return (
        row.get("student_id"),
        row.get("assignment_name"),
        row.get("question_name"),
        len(row.get("curr_problem_prior_submissions") or []),
    )

if os.path.exists(script_args.output_data_path):
    with open(script_args.output_data_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                existing_keys.add(generation_key(entry))
            except:
                continue

print(f"[INFO] Skipping {len(existing_keys)} existing generated rows")

dataset = dataset.filter(
    lambda x: generation_key(x) not in existing_keys
)

print(f"[INFO] Running generation on {len(dataset)} new examples")



# ---------------------
# Print and run one example if index is given
# ---------------------
if args.index is not None:
    sample = dataset[args.index]
    print("\n======== INPUT ========\n")
    print(sample["input"])
    print("\n======== GROUND TRUTH OUTPUT ========\n")
    print(sample["output"])

    if script_args.model_type == "local":
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(script_args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inputs = tokenizer(sample["input"], return_tensors="pt", padding=True, truncation=True, max_length=script_args.max_seq_length).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=script_args.max_new_tokens)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_output = decoded[len(sample["input"]):]
        print("\n======== MODEL OUTPUT ========\n")
        print(model_output.strip())

    elif script_args.model_type in {"api", "openai"}:
        client_kwargs = {"api_key": getattr(script_args, "api_key", None) or os.getenv("OPENAI_API_KEY")}
        api_base = getattr(script_args, "api_base", None) or os.getenv("OPENAI_BASE_URL")
        if api_base:
            client_kwargs["base_url"] = api_base
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=script_args.model_id,
            messages=[
                {"role": "system", "content": script_args.system_prompt},
                {"role": "user", "content": sample["input"]}
            ],
            max_completion_tokens=script_args.max_new_tokens,
            temperature=script_args.temperature,
            top_p=script_args.top_p,
            reasoning_effort=getattr(script_args, "reasoning_effort", "minimal"),
        )
        model_output = response.choices[0].message.content
        print("\n======== MODEL OUTPUT ========\n")
        print(model_output.strip())

    elif script_args.model_type == "vllm":
        _extra_body = {}

        def process_example_vllm(example):
            try:
                response = completion(
                    model=f"hosted_vllm/{script_args.model_id}",
                    messages=[
                        {"role": "system", "content": script_args.system_prompt},
                        {"role": "user", "content": example["input"]}
                    ],
                    api_base=script_args.api_base,
                    max_completion_tokens=script_args.max_new_tokens,
                    temperature=script_args.temperature,
                    top_p=script_args.top_p,
                    top_k=script_args.top_k,
                    min_p=script_args.min_p,
                    repetition_penalty=getattr(script_args, "repetition_penalty", 1.0),
                    stop=getattr(script_args, "stop", None),
                    extra_body=_extra_body or None,
                )
                choice = response["choices"][0]["message"] if response and response.get("choices") else {}
                reasoning = (choice.get("reasoning_content") or "").strip()
                content = (choice.get("content") or "").strip()
                output_text = f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content

            except Exception as e:
                print(f"Error: {e}")
                output_text = f"ERRORED ({e})"

            return {
                **{k: v for k, v in example.items() if k not in ("input", "output")},
                "input": example["input"],
                "output_synthetic": output_text,
                "output_gt": example["output"],
            }

        output_dir = os.path.dirname(os.path.abspath(script_args.output_data_path))
        os.makedirs(output_dir, exist_ok=True)

        with open(script_args.output_data_path, "a", encoding="utf-8") as f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                for result in tqdm(executor.map(process_example_vllm, dataset), total=len(dataset)):
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()


    exit(0)

# ---------------------
# Main Generation
# ---------------------
results = []

if script_args.model_type == "local":
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(script_args.model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sampling_params = {
        "do_sample": script_args.do_sample,
        "temperature": script_args.temperature,
        "top_p": script_args.top_p,
        "top_k": script_args.top_k,
        "min_p": script_args.min_p,
        "max_new_tokens": script_args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": getattr(script_args, "repetition_penalty", 1.0),
    }

    stop_strings = getattr(script_args, "stop", None)
    if stop_strings:
        class StopOnStrings(StoppingCriteria):
            def __init__(self, stop_ids_list):
                self.stop_ids_list = stop_ids_list
            def __call__(self, input_ids, scores, **kwargs):
                for stop_ids in self.stop_ids_list:
                    if input_ids[0, -len(stop_ids):].tolist() == stop_ids:
                        return True
                return False
        stop_ids_list = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
        sampling_params["stopping_criteria"] = StoppingCriteriaList([StopOnStrings(stop_ids_list)])

    output_dir = os.path.dirname(os.path.abspath(script_args.output_data_path))
    os.makedirs(output_dir, exist_ok=True)
    f = open(script_args.output_data_path, "a", encoding="utf-8")

    for start_idx in tqdm(range(0, len(dataset), script_args.batch_size)):
        batch = dataset[start_idx:(start_idx + script_args.batch_size)]
        input_texts = batch["input"]
        output_texts_gt = batch["output"]

        inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True, max_length=script_args.max_seq_length).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **sampling_params)

        decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for idx, full_text in enumerate(decoded_outputs):
            output_text = full_text[len(input_texts[idx]):]
            output_entry = {
                **{k: v[idx] for k, v in batch.items() if k not in ("input", "output")},
                "input": input_texts[idx],
                "output_synthetic": output_text,
                "output_gt": output_texts_gt[idx],
            }
            f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
            f.flush()

    f.close()



elif script_args.model_type in {"api", "openai"}:
    client_kwargs = {"api_key": getattr(script_args, "api_key", None) or os.getenv("OPENAI_API_KEY")}
    api_base = getattr(script_args, "api_base", None) or os.getenv("OPENAI_BASE_URL")
    if api_base:
        client_kwargs["base_url"] = api_base
    client = OpenAI(**client_kwargs)

    def process_example_api(example, max_retries=5):
        num_retries = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=script_args.model_id,
                    messages=[
                        {"role": "system", "content": script_args.system_prompt},
                        {"role": "user", "content": example["input"]}
                    ],
                    max_completion_tokens=script_args.max_new_tokens,
                    temperature=script_args.temperature,
                    top_p=script_args.top_p,
                    reasoning_effort=getattr(script_args, "reasoning_effort", "minimal"),
                )
                return {
                    **{k: v for k, v in example.items() if k not in ("input", "output")},
                    "input": example["input"],
                    "output_synthetic": response.choices[0].message.content,
                    "output_gt": example["output"],
                }

            except Exception as e:
                print(f"[ERROR] API call failed: {e}")
                if num_retries < max_retries:
                    num_retries += 1
                    time.sleep(30)
                else:
                    return {
                        "input": example.get("input", "MISSING"),
                        "output_synthetic": f"ERRORED ({e})",
                        "output_gt": example.get("output", "MISSING"),
                    }


    output_dir = os.path.dirname(os.path.abspath(script_args.output_data_path))
    os.makedirs(output_dir, exist_ok=True)

    with open(script_args.output_data_path, "a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for result in tqdm(executor.map(process_example_api, dataset), total=len(dataset)):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush() 


elif script_args.model_type == "vllm":
    _extra_body = {
        "chat_template_kwargs": {
            "enable_thinking": getattr(script_args, "enable_thinking", False)
        }
    }
    if getattr(script_args, "stop", None):
        _extra_body["include_stop_str_in_output"] = True
        _extra_body["stop"] = script_args.stop

    def process_example_vllm(example):
        try:
            response = completion(
                model=f"hosted_vllm/{script_args.model_id}",
                messages=[
                    {"role": "system", "content": script_args.system_prompt},
                    {"role": "user", "content": example["input"]}
                ],
                api_base=script_args.api_base,
                max_completion_tokens=script_args.max_new_tokens,
                temperature=script_args.temperature,
                top_p=script_args.top_p,
                top_k=script_args.top_k,
                min_p=script_args.min_p,
                extra_body=_extra_body or None,
            )
            choice = response["choices"][0]["message"] if response and response.get("choices") else {}
            reasoning = (choice.get("reasoning_content") or "").strip()
            content = (choice.get("content") or "").strip()
            output_text = f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content

        except Exception as e:
            print(f"Error: {e}")
            output_text = f"ERRORED ({e})"

        return {
            **{k: v for k, v in example.items() if k not in ("input", "output")},
            "input": example["input"],
            "output_synthetic": output_text,
            "output_gt": example["output"],
        }

    output_dir = os.path.dirname(os.path.abspath(script_args.output_data_path))
    os.makedirs(output_dir, exist_ok=True)

    with open(script_args.output_data_path, "a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for result in tqdm(executor.map(process_example_vllm, dataset), total=len(dataset)):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


else:
    raise ValueError(f"Model type {script_args.model_type} not supported")

print(f"Results saved to {script_args.output_data_path}")
