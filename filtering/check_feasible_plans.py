"""
Approach feasibility check (filtering stage 3), Qwen3-4B solver + Qwen3-4B judge.

For every (problem, approach plan) pair a vLLM-served solver model is asked
to solve the problem *following that plan*. Each rollout is verified with
math-verify first and, when that fails, with the LLM verifier (Qwen3-4B
judge). Rollouts are progressive (1, 2, 4, ... up to --n_rollouts) and a
plan is *feasible* as soon as one rollout is verified correct.

Outputs
  <output_file>            one record per (problem, approach, rollout)
  <output_file>_verified   same records with the `verified` flag
  feasible_plans/<name>    input records restricted to feasible approaches,
                           keeping problems with >= 3 feasible approaches
"""

import json
import argparse
import os
import copy
import queue
import threading
from openai import OpenAI
from tqdm import tqdm
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from math_verify import parse, verify

SOLVER_SYSTEM_PROMPT = """
You are a plan-faithful solver.

<INPUTS>
- PROBLEM: the problem statement.
- SELECTED_PLAN: one approach with title/core_idea/assumptions and a numbered list of plan steps (≈3–8 items).

<GOAL>
Produce a complete, correct solution by instantiating and expanding SELECTED_PLAN without changing its core mechanism.

<FIDELITY RULES>
- Execute steps in order following the plan
- Keep strictly to SELECTED_PLAN; do not introduce alternative approaches.
- If any step is invalid/underspecified, STOP and output a brief diagnosis starting with "PLAN MISMATCH:" and request a revised plan. Do not improvise a new strategy.

<OUTPUT FORMAT>
Provide your reasoning and work through the problem step by step, following the plan.
Please reason step by step, and put your final answer within \\boxed{{}}.
"""

def extract_boxed_answer(solution: str):
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return None
    end_idx = idx + 7
    depth = 1
    while end_idx < len(solution):
        if solution[end_idx] == "{":
            depth += 1
        elif solution[end_idx] == "}":
            depth -= 1
        if depth == 0:
            break
        end_idx += 1
           
    ret = solution[idx + 7:end_idx]
    return ret    

def build_solver_prompt(problem: str, plan: List[str]):
    usr_msg = f"Problem: {problem}\n"
    for i, step in enumerate(plan):
        usr_msg += f"step {i+1}: {step}\n"
    return [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": usr_msg}
    ]

def build_solver_prompt_for_client(problem: str, plan: List[str], tokenizer: AutoTokenizer):
    """Build prompt string for completion API with thinking mode enabled"""
    usr_msg = f"Problem: {problem}\n"
    for i, step in enumerate(plan):
        usr_msg += f"step {i+1}: {step}\n"
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": usr_msg}
    ]
    # Apply chat template with thinking mode enabled
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=True
    )
    return prompt

def query_solver_batch(client: OpenAI, model_name: str, prompts: List[str], n_rollout: int) -> List[List[str]]:
    """Query solver using batch completion API"""
    response = client.completions.create(
        model=model_name,
        prompt=prompts,
        n=n_rollout,
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
        presence_penalty=1.0,
        extra_body={"top_k": 20},
    )
    results = [[] for _ in range(len(prompts))]
    if response.choices:
        if hasattr(response.choices[0], "prompt_index"):
            for choice in response.choices:
                results[choice.prompt_index].append(choice.text)
        elif len(response.choices) == len(prompts) * n_rollout:
            idx = 0
            for p_idx in range(len(prompts)):
                results[p_idx] = [response.choices[idx + j].text for j in range(n_rollout)]
                idx += n_rollout
        else:
            for choice in response.choices:
                results[0].append(choice.text)
    return results

def filter_healthy_base_urls(base_urls: List[str], timeout: float = 10.0) -> List[str]:
    """Filter healthy base URLs by checking if they respond to health check"""
    healthy = []
    for url in base_urls:
        client = OpenAI(api_key="EMPTY", base_url=url, timeout=timeout)
        try:
            client.models.list()
            healthy.append(url)
        except Exception as exc:
            print(f"Skipping solver client {url}: health check failed ({exc})")
    return healthy

def generate_with_clients(
    clients: List[OpenAI],
    model_name: str,
    prompts: List[str],
    n_rollout: int,
    batch_size: int,
    progress_desc: str | None = None,
    client_names: List[str] | None = None,
    log_client_usage: bool = True,
    on_batch_results: Callable[[List[int], List[List[str]]], None] | None = None,
) -> List[List[str]]:
    """Generate solutions using multiple clients with batching"""
    if not clients:
        raise ValueError("No solver clients available. Provide at least one --base_url.")
    results: List[List[str] | None] = [None] * len(prompts)
    batches: List[tuple[List[int], List[str]]] = []
    for start in range(0, len(prompts), batch_size):
        batch_indices = list(range(start, min(start + batch_size, len(prompts))))
        batch_prompts = [prompts[i] for i in batch_indices]
        batches.append((batch_indices, batch_prompts))

    total_batches = len(batches)
    progress = tqdm(total=total_batches, desc=progress_desc, leave=False) if progress_desc else None
    work_queue: queue.Queue[tuple[List[int], List[str]]] = queue.Queue()
    for batch in batches:
        work_queue.put(batch)

    errors: List[Exception] = []
    errors_lock = threading.Lock()

    client_batch_counts = [0] * len(clients)

    def worker(client: OpenAI, client_idx: int):
        local_count = 0
        while True:
            try:
                batch_indices, batch_prompts = work_queue.get_nowait()
            except queue.Empty:
                break
            try:
                batch_results = query_solver_batch(client, model_name, batch_prompts, n_rollout)
                for idx, output in zip(batch_indices, batch_results):
                    results[idx] = output
                if on_batch_results is not None:
                    on_batch_results(batch_indices, batch_results)
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)
            finally:
                local_count += 1
                if progress is not None:
                    progress.update(1)
                work_queue.task_done()
        client_batch_counts[client_idx] = local_count

    num_workers = len(clients)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(worker, client, idx)
            for idx, client in enumerate(clients)
        ]
        for future in futures:
            future.result()

    if errors:
        raise errors[0]
    if progress is not None:
        progress.close()
    if log_client_usage:
        names = client_names or [f"client_{i}" for i in range(len(clients))]
        usage = ", ".join(
            f"{name}={count} batches" for name, count in zip(names, client_batch_counts)
        )
        print(f"Solver usage: {usage}")

    return [r or [] for r in results]

def query_openai_for_verification(client, verifier_model_name, answer: str, pred: str, max_retries=3):
    # first, use math_verify to verify the answer
    try:
        pred_ans = parse(pred)
        gt_ans = parse(answer)
        verified = verify(gt_ans, pred_ans)
        if verified:
            return True
    except:
        pass 
    
    system_prompt = f"""
    You are a math expert.
    You are given a golden answer and a predicted answer from a solver.
    You need to verify if the predicted answer is correct.
    Only output "correct" or "incorrect".
    """
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Golden answer: 540, Predicted answer: The total number of ways the cars can stack up so that all three lanes are occupied is calculated to be 750."},
        {"role": "assistant", "content": "incorrect"},
        {"role": "user", "content": "Golden answer: 3, Predicted answer: The ratio \\\\frac{A C}{A E} = 3."},
        {"role": "assistant", "content": "correct"},
        {"role": "user", "content": f"Golden answer: {answer}, Predicted answer: {pred}"}
    ]
    response = client.chat.completions.create(
        model = verifier_model_name,
        messages = message,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        max_completion_tokens=16,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    )
    return response.choices[0].message.content.lower() == "correct"

def verify_solutions(
    verifier_client: OpenAI,
    verifier_model_name: str,
    pair: Dict[str, Any],
    outputs: List[str],
    n_verifier_workers: int = 4,
) -> tuple[List[Dict[str, Any]], bool]:
    """Verify solutions using verifier client with concurrent workers"""
    gt_parsed = None
    try:
        gt_parsed = parse(pair["answer"])
    except Exception:
        gt_parsed = None

    def verify_single_solution(r_idx: int, output: str) -> tuple[int, Dict[str, Any]]:
        solution = output
        pred = extract_boxed_answer(solution) or ""
        verified = False
        
        # First try math_verify
        if gt_parsed is not None:
            try:
                pred_ans = parse(pred)
                verified = verify(gt_parsed, pred_ans)
            except Exception:
                verified = False
        
        # If not verified, use LLM verifier
        if not verified:
            try:
                verified = query_openai_for_verification(
                    verifier_client,
                    verifier_model_name,
                    pair["answer"],
                    pred,
                )
            except Exception:
                verified = False
        
        rollout_index = pair["attempt_offset"] + r_idx
        record = {
            "item_index": pair["item_index"],
            "approach_index": pair["approach_index"],
            "rollout_index": rollout_index,
            "problem": pair["problem"],
            "plan": pair["plan"],
            "solution": solution,
            "answer": pair["answer"],
            "verified": verified,
        }
        return r_idx, record
    
    # Process verifications concurrently
    records_dict = {}
    any_verified = False
    
    with ThreadPoolExecutor(max_workers=n_verifier_workers) as executor:
        futures = [
            executor.submit(verify_single_solution, r_idx, output)
            for r_idx, output in enumerate(outputs)
        ]
        for future in futures:
            r_idx, record = future.result()
            records_dict[r_idx] = record
            if record["verified"]:
                any_verified = True
    
    # Sort records by rollout index
    records = [records_dict[i] for i in range(len(outputs))]
    
    return records, any_verified

def build_rollout_batches(n_rollouts: int, base_batch_size: int) -> List[int]:
    """Build progressive batch sizes: 1, 2, 4, 8, ... up to n_rollouts"""
    if n_rollouts <= 0:
        return []
    max_round_rollouts = max(1, min(n_rollouts, 32))
    round_sizes = []
    round_size = base_batch_size
    while round_size <= max_round_rollouts:
        round_sizes.append(round_size)
        round_size *= 2
    return round_sizes

def main(args):
    model_name = args.model_name
    n_rollouts = args.n_rollouts
    rollout_batch_size = args.rollout_batch_size
    input_file = args.input_file
    output_file = args.output_file
    verifier_model_name = args.verifier_model_name
    verifier_base_url = args.verifier_base_url
    solver_batch_size = max(1, args.solver_batch_size)
    tokenizer_name = args.tokenizer_name or model_name
    max_token_length = args.max_token_length
    n_verifier_workers = args.n_verifier_workers
    base_urls = args.base_url or []

    if not base_urls:
        raise ValueError("Provide at least one --base_url for the solver server.")
    
    healthy_urls = filter_healthy_base_urls(base_urls)
    if not healthy_urls:
        raise ValueError("No healthy solver servers found after health check.")
    if len(healthy_urls) < len(base_urls):
        print(f"Using {len(healthy_urls)} healthy solver clients (out of {len(base_urls)}).")
    
    n_solver_workers = args.n_solver_workers
    if n_solver_workers > 0:
        # Replicate clients across healthy URLs to reach desired worker count
        solver_clients = []
        for i in range(n_solver_workers):
            url = healthy_urls[i % len(healthy_urls)]
            solver_clients.append(OpenAI(api_key="EMPTY", base_url=url, timeout=3600))
    else:
        solver_clients = [OpenAI(api_key="EMPTY", base_url=url, timeout=3600) for url in healthy_urls]
    verifier_client = OpenAI(api_key="EMPTY", base_url=verifier_base_url, timeout=600)

    print(f"Using {len(solver_clients)} solver clients.")
    print(f"Using {n_verifier_workers} concurrent verifier workers.")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # if output path does not exist, create it
    output_path = os.path.dirname(output_file)
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output path: {output_path}")

    with open(input_file, 'r') as f:
        generated_plans = [json.loads(line) for line in f]
    
    # Check if we need to verify existing solutions
    verified_output_file = output_file.replace(".jsonl", "_verified.jsonl")
    existing_pairs = set()
    
    if os.path.exists(output_file):
        print(f"Found existing output file: {output_file}")
        
        # Load existing solutions
        existing_solutions = []
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    existing_solutions.append(record)
                except:
                    pass
        
        # Check if verified file exists and has valid data
        verified_map = {}
        needs_verification = False
        
        if os.path.exists(verified_output_file):
            print(f"Found verified file: {verified_output_file}")
            with open(verified_output_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "verified" in record:
                            key = (record["item_index"], record["approach_index"], record["rollout_index"])
                            verified_map[key] = record
                        else:
                            needs_verification = True
                    except:
                        needs_verification = True
            
            # Check if all existing solutions have verified information
            for sol in existing_solutions:
                key = (sol["item_index"], sol["approach_index"], sol["rollout_index"])
                if key not in verified_map:
                    needs_verification = True
                    break
        else:
            needs_verification = True
            print(f"Verified file not found. Will verify existing solutions.")
        
        # If verification is needed, verify all existing solutions
        if needs_verification:
            print(f"Verifying {len(existing_solutions)} existing solutions...")
            
            # Collect all verified records (both existing and new)
            all_verified_records = []
            
            # Group solutions by (item_index, approach_index) for batch verification
            solutions_to_verify = []
            for sol in existing_solutions:
                key = (sol["item_index"], sol["approach_index"], sol["rollout_index"])
                
                # If already verified, keep it
                if key in verified_map:
                    all_verified_records.append(verified_map[key])
                    continue
                
                # Skip verification for skipped solutions
                if sol.get("solution") == "[SKIPPED: Prompt too long]":
                    verified_record = dict(sol)
                    verified_record["verified"] = False
                    verified_map[key] = verified_record
                    all_verified_records.append(verified_record)
                    continue
                
                solutions_to_verify.append(sol)
            
            # Verify solutions that need verification using multiple workers
            if solutions_to_verify:
                print(f"Need to verify {len(solutions_to_verify)} solutions using {n_verifier_workers} workers...")
                
                def verify_single_existing_solution(sol: Dict[str, Any]) -> Dict[str, Any]:
                    pred = extract_boxed_answer(sol["solution"]) or ""
                    
                    # First try math_verify
                    verified = False
                    try:
                        gt_parsed = parse(sol["answer"])
                        pred_ans = parse(pred)
                        verified = verify(gt_parsed, pred_ans)
                    except Exception:
                        verified = False
                    
                    # If not verified, use LLM verifier
                    if not verified:
                        try:
                            verified = query_openai_for_verification(
                                verifier_client,
                                verifier_model_name,
                                sol["answer"],
                                pred,
                            )
                        except Exception:
                            verified = False
                    
                    verified_record = dict(sol)
                    verified_record["verified"] = verified
                    return verified_record
                
                # Process verifications in parallel
                with ThreadPoolExecutor(max_workers=n_verifier_workers) as executor:
                    futures = [
                        executor.submit(verify_single_existing_solution, sol)
                        for sol in solutions_to_verify
                    ]
                    
                    for future in tqdm(futures, desc="Verifying solutions"):
                        verified_record = future.result()
                        all_verified_records.append(verified_record)
                        
                        key = (verified_record["item_index"], verified_record["approach_index"], verified_record["rollout_index"])
                        verified_map[key] = verified_record
            
            # Save all verified results (overwrite the file)
            print(f"Saving {len(all_verified_records)} verified results to {verified_output_file}")
            with open(verified_output_file, 'w', encoding="utf-8") as f:
                for record in all_verified_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # Build existing_pairs from verified_map
        for key, record in verified_map.items():
            if record.get("verified") is True:
                existing_pairs.add((record["item_index"], record["approach_index"]))
        
        print(f"Found {len(existing_pairs)} pairs with verified solutions.")
    
    # Build pending pairs: (item, approach) combinations
    # Also track pairs that are too long to process
    pending_pairs = []
    too_long_pairs = []
    
    for item in generated_plans:
        for app_idx, app in enumerate(item["response"]["approaches"]):
            key = (item["index"], app_idx)
            if key in existing_pairs:
                continue
            plan_list = app["plan"]
            prompt = build_solver_prompt_for_client(item["problem"], plan_list, tokenizer)
            
            if len(tokenizer.tokenize(prompt)) > max_token_length:
                # Mark as too long - will be recorded as unfeasible
                too_long_pairs.append({
                    "item_index": item["index"],
                    "approach_index": app_idx,
                    "problem": item["problem"],
                    "plan": plan_list,
                    "answer": item["answer"],
                })
                continue
            
            pending_pairs.append({
                "item_index": item["index"],
                "approach_index": app_idx,
                "problem": item["problem"],
                "plan": plan_list,
                "answer": item["answer"],
                "prompt": prompt,
                "attempt_offset": 0,
            })

    print(f"Found {len(existing_pairs)} existing pairs. Processing {len(pending_pairs)} pairs.")
    print(f"Skipping {len(too_long_pairs)} pairs (prompt too long).")

    # Build round sizes: 1, 2, 4, 8, ...
    round_sizes = build_rollout_batches(n_rollouts, rollout_batch_size)
    print(f"Round sizes: {round_sizes}")

    mode = "a" if os.path.exists(output_file) else "w"
    
    with open(output_file, mode, encoding="utf-8") as out_f:
        # Write records for pairs that are too long (mark as unfeasible)
        for pair in too_long_pairs:
            record = {
                "item_index": pair["item_index"],
                "approach_index": pair["approach_index"],
                "rollout_index": 0,
                "problem": pair["problem"],
                "plan": pair["plan"],
                "solution": "[SKIPPED: Prompt too long]",
                "answer": pair["answer"],
                "verified": False,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
        
        verification_pool_size = max(1, n_verifier_workers)
        with ThreadPoolExecutor(max_workers=verification_pool_size) as verification_executor:
            unresolved = pending_pairs
            for round_size in round_sizes:
                if not unresolved:
                    break

                print(f"\n--- Round: generating {round_size} rollout(s) per pair ---")
                next_unresolved: List[Dict[str, Any]] = []

                prompts = [pair["prompt"] for pair in unresolved]
                print(f"Generating {len(prompts)} prompts with {round_size} samples each...")

                state_lock = threading.Lock()
                pending_verifications: List[Any] = []
                pv_lock = threading.Lock()

                # Bind round-local variables explicitly so the closure doesn't
                # pick up the next iteration's rebinding.
                round_size_local = round_size
                unresolved_local = unresolved
                next_unresolved_local = next_unresolved

                def verify_and_record(pair: Dict[str, Any], output: List[str]) -> None:
                    records, any_verified = verify_solutions(
                        verifier_client=verifier_client,
                        verifier_model_name=verifier_model_name,
                        pair=pair,
                        outputs=output,
                        n_verifier_workers=1,
                    )
                    with state_lock:
                        for record in records:
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        if not any_verified:
                            pair["attempt_offset"] += round_size_local
                            next_unresolved_local.append(pair)

                def handle_batch(batch_indices: List[int], batch_outputs: List[List[str]]) -> None:
                    for idx, output in zip(batch_indices, batch_outputs):
                        pair = unresolved_local[idx]
                        fut = verification_executor.submit(verify_and_record, pair, output)
                        with pv_lock:
                            pending_verifications.append(fut)

                generate_with_clients(
                    solver_clients,
                    model_name=model_name,
                    prompts=prompts,
                    n_rollout=round_size,
                    batch_size=solver_batch_size,
                    progress_desc=f"Generating (n={round_size})",
                    client_names=healthy_urls,
                    on_batch_results=handle_batch,
                )

                # Drain all in-flight verifications before advancing to the next round.
                for fut in pending_verifications:
                    fut.result()

                print(f"Resolved: {len(unresolved) - len(next_unresolved)}, Remaining: {len(next_unresolved)}")
                unresolved = next_unresolved

    if unresolved:
        print(f"\nNo correct solution found for {len(unresolved)} pairs after all rounds.")
    
    # Load all results for feasibility checking
    with open(verified_output_file, 'r') as f:
        results = [json.loads(line) for line in f]
    
    app_per_problem = 4
    rollout_per_approach = n_rollouts
    
    # check feasibility
    check_feasibility(results, app_per_problem, rollout_per_approach)
    
    # save the filtered results
    output_path = os.path.dirname(output_file)
    feasible_plans_folder = os.path.join(output_path, "feasible_plans")
    os.makedirs(feasible_plans_folder, exist_ok=True)
    output_file_filtered = os.path.join(feasible_plans_folder, os.path.basename(output_file))
    feasible_plans_filter(generated_plans, results, app_per_problem, rollout_per_approach, output_file_filtered)
            
def check_feasibility(results: List[dict], app_per_problem: int, rollout_per_approach: int):
    # check the number of feasible approaches, compared to the total number of approaches
    # also, report the number of problems that have more than three feasible approaches

    # Group by item_index
    feasibility_map = {}
    for res in results:
        idx = res.get("item_index")
        if idx is not None:
            if idx not in feasibility_map:
                feasibility_map[idx] = []
            feasibility_map[idx].append(res)

    feasible_approaches = 0
    problems_more_than_three_feasible = 0
    
    total_problems = len(feasibility_map)
    total_approaches = total_problems * app_per_problem
    
    for idx, problem_results in feasibility_map.items():
        app_groups = {}
        for r in problem_results:
            a_idx = r.get("approach_index")
            if a_idx is not None:
                if a_idx not in app_groups:
                    app_groups[a_idx] = []
                app_groups[a_idx].append(r)
        
        problem_feasible = 0
        for j in range(app_per_problem):
            rollouts = app_groups.get(j, [])
            if any(r.get("verified", False) for r in rollouts):
                problem_feasible += 1
        
        feasible_approaches += problem_feasible
        if problem_feasible >= 3:
            problems_more_than_three_feasible += 1
            
    print(f"Total problems: {total_problems}")
    print(f"Total approaches: {total_approaches}")
    print(f"Feasible approaches: {feasible_approaches}")
    print(f"Problems with more than three feasible approaches: {problems_more_than_three_feasible}")
    
    ratio_feasible = feasible_approaches / total_approaches if total_approaches > 0 else 0.0
    ratio_problems = problems_more_than_three_feasible / total_problems if total_problems > 0 else 0.0
    
    print(f"Feasibility ratio: {ratio_feasible}")
    print(f"Problems with more than three feasible approaches ratio: {ratio_problems}")
    
    return ratio_feasible, ratio_problems
            
            
def feasible_plans_filter(original_results: List[dict], feasiblity_results: List[dict], app_per_problem: int, rollout_per_approach: int, output_file: str, min_feasible_apps: int = 3):
    # to the output file, only keep the feasible plans
    # that is, only keep the plans that have at least one correct solution
    
    # 1. Group feasibility results by item_index
    feasibility_map = {}
    for res in feasiblity_results:
        idx = res.get("item_index")
        if idx is not None:
            if idx not in feasibility_map:
                feasibility_map[idx] = []
            feasibility_map[idx].append(res)

    filtered_results = []
    
    # 2. Process each original item
    for original_item in original_results:
        idx = original_item.get("index")
        
        if idx not in feasibility_map:
            continue
            
        problem_results = feasibility_map[idx]
        
        # Group by approach_index to be safe against ordering
        app_groups = {}
        for r in problem_results:
            a_idx = r.get("approach_index")
            if a_idx is not None:
                if a_idx not in app_groups:
                    app_groups[a_idx] = []
                app_groups[a_idx].append(r)

        approach_feasible = [False] * app_per_problem
        for j in range(app_per_problem):
            rollouts = app_groups.get(j, [])
            if any(r.get("verified", False) for r in rollouts):
                approach_feasible[j] = True
        
        if sum(approach_feasible) >= min_feasible_apps:
            # Use deepcopy to avoid modifying original data
            new_item = copy.deepcopy(original_item)
            valid_approaches = []
            current_approaches = new_item["response"]["approaches"]
            
            # Filter approaches
            for a in range(len(current_approaches)):
                if a < len(approach_feasible) and approach_feasible[a]:
                    valid_approaches.append(current_approaches[a])
            
            new_item["response"]["approaches"] = valid_approaches
            filtered_results.append(new_item)
            
    # import pdb; pdb.set_trace()
            
    print(f"Filtered {len(filtered_results)} problems from {len(original_results)} inputs.")

    with open(output_file, "w", encoding="utf-8") as f:
        for data in filtered_results:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_url",
        type=str,
        action="append",  
        required=True
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--n_rollouts", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=1, help="Initial batch size for progressive rollouts")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--solver-batch-size", type=int, default=64)
    parser.add_argument("--verifier-model-name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--verifier-base-url", type=str, required=True)
    parser.add_argument("--max-token-length", type=int, default=8192, help="Maximum token length for prompts")
    parser.add_argument("--n-verifier-workers", type=int, default=4, help="Number of concurrent verifier workers sharing one client")
    parser.add_argument("--n-solver-workers", type=int, default=0, help="Number of concurrent solver workers (0 = one per client URL)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
